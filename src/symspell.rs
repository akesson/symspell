use std::cmp;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::i64;
use std::io::{BufRead, BufReader};
use std::path::Path;

use composition::Composition;
use edit_distance::{DistanceAlgorithm, EditDistance};
use string_strategy::StringStrategy;
use suggestion::Suggestion;

#[derive(Eq, PartialEq, Debug)]
pub enum Verbosity {
    Top,
    Closest,
    All,
}

#[derive(Builder, PartialEq)]
pub struct SymSpell<T: StringStrategy> {
    /// Maximum edit distance for doing lookups.
    #[builder(default = "2")]
    max_dictionary_edit_distance: i64,
    /// The length of word prefixes used for spell checking.
    #[builder(default = "7")]
    prefix_length: i64,
    /// The minimum frequency count for dictionary words to be considered correct spellings.
    #[builder(default = "1")]
    count_threshold: i64,

    //// number of all words in the corpus used to generate the
    //// frequency dictionary. This is used to calculate the word
    //// occurrence probability p from word counts c : p=c/N. N equals
    //// the sum of all counts c in the dictionary only if the
    //// dictionary is complete, but not if the dictionary is
    //// truncated or filtered
    #[builder(default = "1_024_908_267_229", setter(skip))]
    corpus_word_count: i64,

    #[builder(default = "0", setter(skip))]
    max_length: i64,
    #[builder(default = "HashMap::new()", setter(skip))]
    deletes: HashMap<u64, Vec<String>>,
    #[builder(default = "HashMap::new()", setter(skip))]
    words: HashMap<String, i64>,
    #[builder(default = "DistanceAlgorithm::Damerau")]
    distance_algorithm: DistanceAlgorithm,
    #[builder(default = "T::new()", setter(skip))]
    string_strategy: T,
}

impl<T: StringStrategy> Default for SymSpell<T> {
    fn default() -> SymSpell<T> {
        SymSpellBuilder::default().build().unwrap()
    }
}

impl<T: StringStrategy> SymSpell<T> {
    /// Load multiple dictionary entries from a file of word/frequency count pairs.
    ///
    /// # Arguments
    ///
    /// * `corpus` - The path+filename of the file.
    /// * `term_index` - The column position of the word.
    /// * `count_index` - The column position of the frequency count.
    /// * `separator` - Separator between word and frequency
    pub fn load_dictionary(
        &mut self,
        corpus: &str,
        term_index: i64,
        count_index: i64,
        separator: &str,
    ) -> bool {
        if !Path::new(corpus).exists() {
            return false;
        }

        let file = File::open(corpus).expect("file not found");
        let sr = BufReader::new(file);

        for (i, line) in sr.lines().enumerate() {
            if i % 50_000 == 0 {
                println!("progress: {}", i);
            }
            let line_str = line.unwrap();
            self.load_dictionary_line(&line_str, term_index, count_index, separator);
        }
        true
    }

    /// Load single dictionary entry from word/frequency count pair.
    ///
    /// # Arguments
    ///
    /// * `line` - word/frequency pair.
    /// * `term_index` - The column position of the word.
    /// * `count_index` - The column position of the frequency count.
    /// * `separator` - Separator between word and frequency
    pub fn load_dictionary_line(
        &mut self,
        line: &str,
        term_index: i64,
        count_index: i64,
        separator: &str,
    ) -> bool {
        let line_parts: Vec<&str> = line.split(separator).collect();
        if line_parts.len() >= 2 {
            // let key = unidecode(line_parts[term_index as usize]);
            let key = self
                .string_strategy
                .prepare(line_parts[term_index as usize]);
            let count = line_parts[count_index as usize].parse::<i64>().unwrap();

            self.create_dictionary_entry(key, count);
        }
        true
    }

    pub fn info(&self) -> String {
        let mut tot_len = 0;
        for (_, vec) in self.deletes.iter() {
            tot_len += vec.len();
        }
        format!(
            "Words: {}, Deletes: {} with avg vec.len: {}",
            self.words.len(),
            self.deletes.len(),
            tot_len / self.deletes.len()
        )
    }
    /// Find suggested spellings for a given input word, using the maximum
    /// edit distance specified during construction of the SymSpell dictionary.
    ///
    /// # Arguments
    ///
    /// * `input` - The word being spell checked.
    /// * `verbosity` - The value controlling the quantity/closeness of the retuned suggestions.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    ///
    /// # Examples
    ///
    /// ```
    /// use symspell::{SymSpell, AsciiStringStrategy, Verbosity};
    ///
    /// let mut symspell: SymSpell<AsciiStringStrategy> = SymSpell::default();
    /// symspell.load_dictionary("data/frequency_dictionary_en_82_765.txt", 0, 1, " ");
    /// symspell.lookup("whatver", Verbosity::Top, 2);
    /// ```
    pub fn lookup(
        &self,
        input: &str,
        verbosity: Verbosity,
        max_edit_distance: i64,
    ) -> Vec<Suggestion> {
        if max_edit_distance > self.max_dictionary_edit_distance {
            panic!("max_edit_distance is bigger than max_dictionary_edit_distance");
        }

        let mut suggestions: Vec<Suggestion> = Vec::new();

        let prep_input = self.string_strategy.prepare(input);
        let input = prep_input.as_str();
        let input_len = self.string_strategy.len(input) as i64;

        if input_len - self.max_dictionary_edit_distance > self.max_length {
            return suggestions;
        }

        let mut hashset1: HashSet<String> = HashSet::new();
        let mut hashset2: HashSet<String> = HashSet::new();

        if self.words.contains_key(input) {
            let suggestion_count = self.words[input];
            suggestions.push(Suggestion::new(input, 0, suggestion_count));

            if verbosity != Verbosity::All {
                return suggestions;
            }
        }

        hashset2.insert(input.to_string());

        let mut max_edit_distance2 = max_edit_distance;
        let mut candidate_pointer = 0;
        let mut candidates = Vec::new();

        let mut input_prefix_len = input_len;

        if input_prefix_len > self.prefix_length {
            input_prefix_len = self.prefix_length;
            candidates.push(
                self.string_strategy
                    .slice(input, 0, input_prefix_len as usize),
            );
        } else {
            candidates.push(input.to_string());
        }

        let distance_comparer = EditDistance::new(self.distance_algorithm.clone());

        while candidate_pointer < candidates.len() {
            let candidate = &candidates.get(candidate_pointer).unwrap().clone();
            candidate_pointer += 1;
            let candidate_len = self.string_strategy.len(candidate) as i64;
            let length_diff = input_prefix_len - candidate_len;

            if length_diff > max_edit_distance2 {
                if verbosity == Verbosity::All {
                    continue;
                }
                break;
            }

            if self.deletes.contains_key(&self.get_string_hash(&candidate)) {
                let dict_suggestions = &self.deletes[&self.get_string_hash(&candidate)];

                for suggestion in dict_suggestions {
                    let suggestion_len = self.string_strategy.len(suggestion) as i64;

                    if suggestion == input {
                        continue;
                    }

                    if (suggestion_len - input_len).abs() > max_edit_distance2
                        || suggestion_len < candidate_len
                        || (suggestion_len == candidate_len && suggestion != candidate)
                    {
                        continue;
                    }

                    let sugg_prefix_len = cmp::min(suggestion_len, self.prefix_length);

                    if sugg_prefix_len > input_prefix_len
                        && sugg_prefix_len - candidate_len > max_edit_distance2
                    {
                        continue;
                    }

                    let distance;

                    if candidate_len == 0 {
                        distance = cmp::max(input_len, suggestion_len);

                        if distance > max_edit_distance2 || hashset2.contains(suggestion) {
                            continue;
                        }
                        hashset2.insert(suggestion.to_string());
                    } else if suggestion_len == 1 {
                        distance = if !input.contains(&self.string_strategy.slice(suggestion, 0, 1))
                        {
                            input_len
                        } else {
                            input_len - 1
                        };

                        if distance > max_edit_distance2 || hashset2.contains(suggestion) {
                            continue;
                        }

                        hashset2.insert(suggestion.to_string());
                    } else if self.has_different_suffix(
                        max_edit_distance,
                        input,
                        input_len,
                        candidate_len,
                        suggestion,
                        suggestion_len,
                    ) {
                        continue;
                    } else {
                        if verbosity != Verbosity::All
                            && !self.delete_in_suggestion_prefix(
                                candidate,
                                candidate_len,
                                suggestion,
                                suggestion_len,
                            )
                        {
                            continue;
                        }

                        if hashset2.contains(suggestion) {
                            continue;
                        }
                        hashset2.insert(suggestion.to_string());

                        distance = distance_comparer.compare(input, suggestion, max_edit_distance2);

                        if distance < 0 {
                            continue;
                        }
                    }

                    if distance <= max_edit_distance2 {
                        let suggestion_count = self.words[suggestion];
                        let si = Suggestion::new(suggestion, distance, suggestion_count);

                        if !suggestions.is_empty() {
                            match verbosity {
                                Verbosity::Closest => {
                                    if distance < max_edit_distance2 {
                                        suggestions.clear();
                                    }
                                }
                                Verbosity::Top => {
                                    if distance < max_edit_distance2
                                        || suggestion_count > suggestions[0].count
                                    {
                                        max_edit_distance2 = distance;
                                        suggestions[0] = si;
                                    }
                                    continue;
                                }
                                _ => (),
                            }
                        }

                        if verbosity != Verbosity::All {
                            max_edit_distance2 = distance;
                        }

                        suggestions.push(si);
                    }
                }
            }

            if length_diff < max_edit_distance && candidate_len <= self.prefix_length {
                if verbosity != Verbosity::All && length_diff >= max_edit_distance2 {
                    continue;
                }

                for i in 0..candidate_len {
                    let delete = self.string_strategy.remove(candidate, i as usize);

                    if !hashset1.contains(&delete) {
                        hashset1.insert(delete.clone());
                        candidates.push(delete);
                    }
                }
            }
        }

        if suggestions.len() > 1 {
            suggestions.sort();
        }

        suggestions
    }

    /// Divides a string into words by inserting missing spaces at the appropriate positions
    ///
    ///
    /// # Arguments
    ///
    /// * `input` - The word being segmented.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    ///
    /// # Examples
    ///
    /// ```
    /// use symspell::{SymSpell, UnicodeStringStrategy, Verbosity};
    ///
    /// let mut symspell: SymSpell<UnicodeStringStrategy> = SymSpell::default();
    /// symspell.load_dictionary("data/frequency_dictionary_en_82_765.txt", 0, 1, " ");
    /// symspell.word_segmentation("itwas", 2);
    /// ```
    pub fn word_segmentation(&self, input: &str, max_edit_distance: i64) -> Composition {
        let input = self.string_strategy.prepare(input);
        let asize = self.string_strategy.len(&input);

        let mut ci: usize = 0;
        let mut compositions: Vec<Composition> = vec![Composition::empty(); asize];

        for j in 0..asize {
            let imax = cmp::min(asize - j, self.max_length as usize);
            for i in 1..=imax {
                let top_prob_log: f64;

                let mut part = self.string_strategy.slice(&input, j, j + i);

                let mut sep_len = 0;
                let mut top_ed: i64 = 0;

                let first_char = self.string_strategy.at(&part, 0).unwrap();
                if first_char.is_whitespace() {
                    part = self.string_strategy.remove(&part, 0);
                } else {
                    sep_len = 1;
                }

                top_ed += part.len() as i64;

                part = part.replace(" ", "");

                top_ed -= part.len() as i64;

                let results = self.lookup(&part, Verbosity::Top, max_edit_distance);

                if !results.is_empty() && results[0].distance == 0 {
                    top_prob_log =
                        (results[0].count as f64 / self.corpus_word_count as f64).log10();
                } else {
                    top_ed += part.len() as i64;
                    top_prob_log = (10.0
                        / (self.corpus_word_count as f64 * 10.0f64.powf(part.len() as f64)))
                    .log10();
                }

                let di = (i + ci) % asize;
                // set values in first loop
                if j == 0 {
                    compositions[i - 1] = Composition {
                        segmented_string: part.to_owned(),
                        distance_sum: top_ed,
                        prob_log_sum: top_prob_log,
                    };
                } else if i as i64 == self.max_length
                    || (((compositions[ci].distance_sum + top_ed == compositions[di].distance_sum)
                        || (compositions[ci].distance_sum + sep_len + top_ed
                            == compositions[di].distance_sum))
                        && (compositions[di].prob_log_sum
                            < compositions[ci].prob_log_sum + top_prob_log))
                    || (compositions[ci].distance_sum + sep_len + top_ed
                        < compositions[di].distance_sum)
                {
                    compositions[di] = Composition {
                        segmented_string: format!("{} {}", compositions[ci].segmented_string, part),
                        distance_sum: compositions[ci].distance_sum + sep_len + top_ed,
                        prob_log_sum: compositions[ci].prob_log_sum + top_prob_log,
                    };
                }
            }
            if j != 0 {
                ci += 1;
            }
            ci = if ci == asize { 0 } else { ci };
        }
        compositions[ci].to_owned()
    }

    fn delete_in_suggestion_prefix(
        &self,
        delete: &str,
        delete_len: i64,
        suggestion: &str,
        suggestion_len: i64,
    ) -> bool {
        if delete_len == 0 {
            return true;
        }
        let suggestion_len = if self.prefix_length < suggestion_len {
            self.prefix_length
        } else {
            suggestion_len
        };
        let mut j = 0;
        for i in 0..delete_len {
            let del_char = self.string_strategy.at(delete, i as isize).unwrap();
            while j < suggestion_len
                && del_char != self.string_strategy.at(suggestion, j as isize).unwrap()
            {
                j += 1;
            }

            if j == suggestion_len {
                return false;
            }
        }
        true
    }

    pub fn create_dictionary_entry(&mut self, key: String, count: i64) -> bool {
        if count < self.count_threshold {
            return false;
        }

        match self.words.get(&key) {
            Some(i) => {
                let updated_count = if i64::MAX - i > count {
                    i + count
                } else {
                    i64::MAX
                };
                self.words.insert(key.clone(), updated_count);
                return false;
            }
            None => {
                self.words.insert(key.clone(), count);
            }
        }

        let key_len = self.string_strategy.len(&key);

        if key_len as i64 > self.max_length {
            self.max_length = key_len as i64;
        }

        let edits = self.edits_prefix(&key);

        for delete in edits {
            let delete_hash = self.get_string_hash(&delete);

            self.deletes
                .entry(delete_hash)
                .and_modify(|e| e.push(key.clone()))
                .or_insert_with(|| vec![key.to_string()]);
        }

        true
    }

    fn edits_prefix(&self, key: &str) -> HashSet<String> {
        let mut hash_set = HashSet::new();

        let key_len = self.string_strategy.len(key) as i64;

        if key_len <= self.max_dictionary_edit_distance {
            hash_set.insert("".to_string());
        }

        if key_len > self.prefix_length {
            let shortened_key = self
                .string_strategy
                .slice(key, 0, self.prefix_length as usize);
            hash_set.insert(shortened_key.clone());
            self.edits(&shortened_key, 0, &mut hash_set);
        } else {
            hash_set.insert(key.to_string());
            self.edits(key, 0, &mut hash_set);
        };

        hash_set
    }

    fn edits(&self, word: &str, edit_distance: i64, delete_words: &mut HashSet<String>) {
        let edit_distance = edit_distance + 1;
        let word_len = self.string_strategy.len(word);

        if word_len > 1 {
            for i in 0..word_len {
                let delete = self.string_strategy.remove(word, i);

                if !delete_words.contains(&delete) {
                    delete_words.insert(delete.clone());

                    if edit_distance < self.max_dictionary_edit_distance {
                        self.edits(&delete, edit_distance, delete_words);
                    }
                }
            }
        }
    }

    fn has_different_suffix(
        &self,
        max_edit_distance: i64,
        input: &str,
        input_len: i64,
        candidate_len: i64,
        suggestion: &str,
        suggestion_len: i64,
    ) -> bool {
        // handles the shortcircuit of min_distance
        // assignment when first boolean expression
        // evaluates to false
        let min = if self.prefix_length - max_edit_distance == candidate_len {
            cmp::min(input_len, suggestion_len) - self.prefix_length
        } else {
            0
        };

        (self.prefix_length - max_edit_distance == candidate_len)
            && (((min - self.prefix_length) > 1)
                && (self
                    .string_strategy
                    .suffix(input, (input_len + 1 - min) as usize)
                    != self
                        .string_strategy
                        .suffix(suggestion, (suggestion_len + 1 - min) as usize)))
            || ((min > 0)
                && (self.string_strategy.at(input, (input_len - min) as isize)
                    != self
                        .string_strategy
                        .at(suggestion, (suggestion_len - min) as isize))
                && ((self
                    .string_strategy
                    .at(input, (input_len - min - 1) as isize)
                    != self
                        .string_strategy
                        .at(suggestion, (suggestion_len - min) as isize))
                    || (self.string_strategy.at(input, (input_len - min) as isize)
                        != self
                            .string_strategy
                            .at(suggestion, (suggestion_len - min - 1) as isize))))
    }

    fn get_string_hash(&self, s: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use string_strategy::UnicodeStringStrategy;

    #[test]
    fn test_word_segmentation() {
        let edit_distance_max = 2;
        let mut sym_spell = SymSpell::<UnicodeStringStrategy>::default();
        sym_spell.load_dictionary("./data/frequency_dictionary_en_82_765.txt", 0, 1, " ");

        let typo = "thequickbrownfoxjumpsoverthelazydog";
        let correction = "the quick brown fox jumps over the lazy dog";
        let result = sym_spell.word_segmentation(typo, edit_distance_max);
        assert_eq!(correction, result.segmented_string);

        let typo = "itwasabrightcolddayinaprilandtheclockswerestrikingthirteen";
        let correction = "it was a bright cold day in april and the clocks were striking thirteen";
        let result = sym_spell.word_segmentation(typo, edit_distance_max);
        assert_eq!(correction, result.segmented_string);

        let typo =
            "itwasthebestoftimesitwastheworstoftimesitwastheageofwisdomitwastheageoffoolishness";
        let correction = "it was the best of times it was the worst of times it was the age of wisdom it was the age of foolishness";
        let result = sym_spell.word_segmentation(typo, edit_distance_max);
        assert_eq!(correction, result.segmented_string);
    }

    #[test]
    fn test_dictionary_creation() {
        let mut sym_spell = SymSpell::<UnicodeStringStrategy>::default();
        sym_spell.create_dictionary_entry("bring".to_string(), 1);
        sym_spell.create_dictionary_entry("blang".to_string(), 3);
        sym_spell.create_dictionary_entry("glong".to_string(), 20);
        let results = sym_spell.lookup("brang", Verbosity::Closest, 2);
        assert_eq!(results[0], Suggestion::new("bing", 1, 1));
        assert_eq!(results[1], Suggestion::new("blang", 1, 3));
    }
}
