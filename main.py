import os
import re
from collections import Counter
import difflib # For similarity scoring

DATASET_DIR = "data/"

def load_all_ayat():
    """
    Loads all verses from text files in the DATASET_DIR.
    Each file is expected to have:
    - Line 1: Book name (e.g., "Genesis.")
    - Line 2: Chapter number (e.g., "Chapter 1.")
    - Lines 3 onwards: Verse content, split into sentences.
    Returns a list of tuples: (book, chapter, verse_number, sentence_text).
    """
    ayat_all = []
    print("Loading biblical texts...")
    
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".txt")]
    for i, filename in enumerate(files, 1):
        print(f"Processing file {i}/{len(files)}: {filename}")
        filepath = os.path.join(DATASET_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()

            if len(lines) < 2:
                print(f"Skipping {filename}: Not enough lines.")
                continue

            kitab = lines[0].strip().replace('.', '')
            
            chapter_line_match = re.match(r"Chapter (\d+)\.", lines[1].strip())
            chapter = int(chapter_line_match.group(1)) if chapter_line_match else None

            if chapter is None:
                print(f"Skipping {filename}: Could not parse chapter from '{lines[1].strip()}'.")
                continue

            content = " ".join([line.strip() for line in lines[2:]])
            # Improved sentence splitting - handles more edge cases
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)

            for i, sentence in enumerate(sentences, 1):
                sentence = sentence.strip()
                if sentence:
                    ayat_all.append((kitab, chapter, i, sentence))
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"Loaded {len(ayat_all)} verses total.")
    return ayat_all

# --- Fixed and Optimized String Searching Algorithms ---

def _build_bad_char_table(pattern):
    """
    Builds the bad character shift table for Boyer-Moore algorithm.
    """
    table = {}
    for i in range(len(pattern)):
        table[pattern[i]] = i
    return table

def boyer_moore_fixed_search(text, pattern):
    """
    Boyer-Moore search implementation.
    Returns True if pattern is found, False otherwise. (Case-insensitive)
    """
    text = text.lower()
    pattern = pattern.lower()
    n = len(text)
    m = len(pattern)
    
    if m == 0: 
        return True
    if n == 0 or m > n: 
        return False

    bad_char_table = _build_bad_char_table(pattern)
    
    shift = 0  # How far we've shifted the pattern
    while shift <= n - m:
        j = m - 1  # Start from the end of pattern
        
        # Match characters from right to left
        while j >= 0 and pattern[j] == text[shift + j]:
            j -= 1
        
        if j < 0:  # Pattern found
            return True
        else:
            # Calculate shift using bad character rule
            bad_char = text[shift + j]
            if bad_char in bad_char_table:
                # Shift pattern so that the bad character in text aligns with its last occurrence in pattern
                shift += max(1, j - bad_char_table[bad_char])
            else:
                # Character not in pattern, shift by pattern length
                shift += j + 1
                
    return False

def search_by_algorithm(ayat_list, keyword, algorithm_name="boyer_moore"):
    """
    Finds verses containing the given keyword using the specified algorithm.
    """
    results = []
    total = len(ayat_list)
    
    print(f"Searching for '{keyword}' using {algorithm_name} algorithm...")
    
    if algorithm_name == "boyer_moore":
        for i, (kitab, chapter, verse, sentence) in enumerate(ayat_list):
            if i % 1000 == 0:  # Progress indicator
                print(f"Progress: {i}/{total} verses processed ({i/total*100:.1f}%)")
            if boyer_moore_fixed_search(sentence, keyword):
                results.append((kitab, chapter, verse, sentence))
    elif algorithm_name == "regex":
        compiled_pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for i, (kitab, chapter, verse, sentence) in enumerate(ayat_list):
            if i % 1000 == 0:
                print(f"Progress: {i}/{total} verses processed ({i/total*100:.1f}%)")
            if compiled_pattern.search(sentence):
                results.append((kitab, chapter, verse, sentence))
    elif algorithm_name == "regex_pattern":  # For complex regex patterns
        compiled_pattern = re.compile(keyword, re.IGNORECASE)
        for i, (kitab, chapter, verse, sentence) in enumerate(ayat_list):
            if i % 1000 == 0:
                print(f"Progress: {i}/{total} verses processed ({i/total*100:.1f}%)")
            if compiled_pattern.search(sentence):
                results.append((kitab, chapter, verse, sentence))
    else:
        raise ValueError("Unsupported algorithm. Choose 'boyer_moore', 'regex', 'regex_pattern', or 'simple_in'.")
    
    print(f"Search complete. Found {len(results)} matches.")
    return results

# --- Optimized Cross-Reference Detection ---

# Expanded and optimized alias mapping
alias_map = {
    "moses": "Exodus", "prophet isaiah": "Isaiah", "isaiah": "Isaiah",
    "jeremiah": "Jeremiah", "the law": "Leviticus", "law of moses": "Deuteronomy",
    "psalms": "Psalms", "book of psalms": "Psalms", "daniel": "Daniel",
    "matthew": "Matthew", "luke": "Luke", "john": "John", "paul": "Romans",
    "peter": "1 Peter", "revelation": "Revelation", "genesis": "Genesis",
    "exodus": "Exodus", "leviticus": "Leviticus", "numbers": "Numbers",
    "deuteronomy": "Deuteronomy", "proverbs": "Proverbs", "ecclesiastes": "Ecclesiastes",
    "song of solomon": "Song of Solomon", "ezekiel": "Ezekiel", "hosea": "Hosea",
    "joel": "Joel", "amos": "Amos", "obadiah": "Obadiah", "jonah": "Jonah",
    "micah": "Micah", "nahum": "Nahum", "habakkuk": "Habakkuk", "zephaniah": "Zephaniah",
    "haggai": "Haggai", "zechariah": "Zechariah", "malachi": "Malachi",
    "romans": "Romans", "1 corinthians": "1 Corinthians", "2 corinthians": "2 Corinthians",
    "galatians": "Galatians", "ephesians": "Ephesians", "philippians": "Philippians",
    "colossians": "Colossians", "1 thessalonians": "1 Thessalonians", "2 thessalonians": "2 Thessalonians",
    "1 timothy": "1 Timothy", "2 timothy": "2 Timothy", "titus": "Titus", "philemon": "Philemon",
    "hebrews": "Hebrews", "james": "James", "1 peter": "1 Peter", "2 peter": "2 Peter",
    "1 john": "1 John", "2 john": "2 John", "3 john": "3 John", "jude": "Jude",
    "acts": "Acts", "ruth": "Ruth", "judges": "Judges", "joshua": "Joshua",
    "1 samuel": "1 Samuel", "2 samuel": "2 Samuel", "1 kings": "1 Kings", "2 kings": "2 Kings",
    "1 chronicles": "1 Chronicles", "2 chronicles": "2 Chronicles", "ezra": "Ezra",
    "nehemiah": "Nehemiah", "esther": "Esther", "job": "Job", "lamentations": "Lamentations",
    "ecclesiastes": "Ecclesiastes"  # Fixed spelling
}

def detect_explicit_biblical_references_optimized(ayat_list, similarity_threshold=0.8):
    """
    Pre-compile regex pattern and use more efficient fuzzy matching.
    """
    # Pre-compile the regex pattern for better performance
    pattern = re.compile(
        r"(?:spoken by|written in|according to|as it is written in|said by|the book of)\s*(?:the\s*)?([A-Z][a-zA-Z\s]+)(?:\s*(?:chapter|ch\.)\s*\d+)?(?:\s*:\s*\d+)?",
        re.IGNORECASE
    )
    
    result = []
    alias_keys = list(alias_map.keys())
    total = len(ayat_list)
    
    print("Detecting biblical cross-references...")
    
    for i, (kitab, chapter, verse, sentence) in enumerate(ayat_list):
        if i % 1000 == 0:
            print(f"Progress: {i}/{total} verses processed ({i/total*100:.1f}%)")
            
        matches = pattern.finditer(sentence)
        for match in matches:
            ref_name_raw = match.group(1).strip()
            
            # Normalize the extracted reference name for comparison
            normalized_ref_name = ref_name_raw.lower()
            if normalized_ref_name.startswith("the "):
                normalized_ref_name = normalized_ref_name[4:]

            # Use get_close_matches for more efficient fuzzy matching
            close_matches = difflib.get_close_matches(
                normalized_ref_name, 
                alias_keys, 
                n=1, 
                cutoff=similarity_threshold
            )
            
            if close_matches:
                best_match_key = close_matches[0]
                # Calculate the actual similarity score
                score = difflib.SequenceMatcher(None, normalized_ref_name, best_match_key).ratio()
                target_kitab = alias_map[best_match_key]
                result.append(((kitab, chapter, verse), target_kitab, ref_name_raw, score))
    
    print(f"Cross-reference detection complete. Found {len(result)} references.")
    return result

def most_common_words_optimized(ayat_list, top_n=10):
    """
    More efficient word counting with better stopword filtering.
    """
    stopwords = {
        "no", "because", "therefore", "most", "after", "let", "what", "there", "behold", 
        "how", "go", "yes", "why", "your", "my", "who", "so", "if", "when", "now", "then", 
        "all", "the", "he", "she", "it", "they", "we", "you", "his", "her", "their", "and", 
        "but", "for", "with", "in", "on", "at", "by", "to", "from", "of", "as", "is", "that", 
        "this", "these", "those", "not", "be", "are", "was", "were", "has", "had", "have", 
        "will", "shall", "may", "can", "do", "did", "does", "unto", "upon", "would", "which", 
        "where", "whom", "him", "her", "them", "us", "our", "an", "a", "or", "said", "thus", 
        "ye", "thy", "thou", "hath", "unto", "came", "went", "out", "into", "up", "down", 
        "from", "through", "before", "after", "while", "wherefore", "verily", "also", "even", 
        "more", "than", "such", "one", "two", "first", "last", "over", "under", "between"
    }
    
    # Pre-compile regex for better performance
    word_pattern = re.compile(r'\b[A-Z][a-zA-Z]+\b')
    
    print("Analyzing word frequencies...")
    counter = Counter()
    
    for i, (_, _, _, sentence) in enumerate(ayat_list):
        if i % 1000 == 0:
            print(f"Progress: {i}/{len(ayat_list)} verses processed ({i/len(ayat_list)*100:.1f}%)")
        
        tokens = word_pattern.findall(sentence)
        filtered_tokens = [
            t.lower() for t in tokens 
            if t.lower() not in stopwords and len(t) > 2
        ]
        counter.update(filtered_tokens)
    
    print("Word frequency analysis complete.")
    return counter.most_common(top_n)

def search_multiple_keywords_optimized(ayat_list, keywords, algorithm_name="boyer_moore"):
    """
    Search for multiple keywords efficiently by scanning each verse only once.
    """
    if algorithm_name not in ["simple_in", "boyer_moore"]:
        # For complex patterns, fall back to individual searches
        all_results = []
        for keyword in keywords:
            results = search_by_algorithm(ayat_list, keyword, algorithm_name)
            all_results.extend(results)
        return list(set(all_results))  # Remove duplicates
    
    print(f"Searching for {len(keywords)} keywords using optimized batch search...")
    results = []
    keywords_lower = [k.lower() for k in keywords] if algorithm_name == "simple_in" else keywords
    
    for i, (kitab, chapter, verse, sentence) in enumerate(ayat_list):
        if i % 1000 == 0:
            print(f"Progress: {i}/{len(ayat_list)} verses processed ({i/len(ayat_list)*100:.1f}%)")
        
        sentence_lower = sentence.lower() if algorithm_name == "simple_in" else sentence
        
        for j, keyword in enumerate(keywords_lower):
            found = False
            if algorithm_name == "simple_in":
                found = keyword in sentence_lower
            elif algorithm_name == "boyer_moore":
                found = boyer_moore_fixed_search(sentence, keyword)
            
            if found:
                results.append((kitab, chapter, verse, sentence))
                break  # Don't need to check other keywords for this verse
    
    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for item in results:
        if item not in seen:
            seen.add(item)
            unique_results.append(item)
    
    print(f"Batch search complete. Found {len(unique_results)} unique matches.")
    return unique_results

def write_results_to_file(all_ayat, output_filename="bible_analysis_results.txt"):
    """
    Writes all analysis results to a formatted text file.
    """
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("BIBLICAL TEXT ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total verses analyzed: {len(all_ayat):,}\n")
        f.write("=" * 80 + "\n\n")

        # Choose algorithm
        chosen_algorithm = "boyer_moore"  # Default algorithm
        f.write(f"Search Algorithm Used: {chosen_algorithm.upper()}\n")
        f.write("-" * 80 + "\n\n")

        # --- Theme 1: Covenant/Promise ---
        f.write("THEME 1: COVENANT/PROMISE\n")
        f.write("=" * 40 + "\n\n")
        
        covenant_keywords = ["covenant", "promise", "oath", "agreement", "blessing", "testament"]
        f.write(f"Keywords searched: {', '.join(covenant_keywords)}\n\n")
        
        covenant_matches = search_multiple_keywords_optimized(all_ayat, covenant_keywords, chosen_algorithm)
        f.write(f"Found {len(covenant_matches)} unique verses for 'Covenant/Promise' theme.\n\n")
        
        # Write sample verses (first 20)
        f.write("SAMPLE VERSES (first 20 matches):\n")
        f.write("-" * 40 + "\n")
        for i, (kitab, chapter, verse, sentence) in enumerate(covenant_matches[:20], 1):
            f.write(f"{i:2d}. {kitab} {chapter}:{verse}\n")
            f.write(f"    {sentence}\n\n")
        
        if len(covenant_matches) > 20:
            f.write(f"... and {len(covenant_matches) - 20} more matches.\n\n")

        # Complex patterns
        covenant_patterns = [
            r"(?:new|everlasting|eternal)\s+covenant",
            r"(?:promise|oath)\s+(?:to|of)\s+(?:Abraham|David|Israel)",
            r"(?:make|establish)\s+a\s+covenant",
        ]
        
        f.write("COMPLEX PATTERN MATCHES:\n")
        f.write("-" * 40 + "\n")
        covenant_pattern_matches = []
        for j, pattern in enumerate(covenant_patterns, 1):
            f.write(f"Pattern {j}: {pattern}\n")
            matches = search_by_algorithm(all_ayat, pattern, "regex_pattern")
            covenant_pattern_matches.extend(matches)
            f.write(f"Matches: {len(matches)}\n")
            
            # Show first 5 matches for each pattern
            for i, (kitab, chapter, verse, sentence) in enumerate(matches[:5], 1):
                f.write(f"  {i}. {kitab} {chapter}:{verse} - {sentence[:100]}{'...' if len(sentence) > 100 else ''}\n")
            if len(matches) > 5:
                f.write(f"  ... and {len(matches) - 5} more.\n")
            f.write("\n")
        
        f.write(f"Total pattern matches: {len(set(covenant_pattern_matches))}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        # --- Theme 2: Sin and Redemption ---
        f.write("THEME 2: SIN AND REDEMPTION/SALVATION\n")
        f.write("=" * 40 + "\n\n")
        
        sin_redemption_keywords = [
            "sin", "transgression", "iniquity", "redeem", "redemption", 
            "save", "salvation", "atonement", "sacrifice", "grace", "faith", "forgive"
        ]
        f.write(f"Keywords searched: {', '.join(sin_redemption_keywords)}\n\n")
        
        sin_redemption_matches = search_multiple_keywords_optimized(all_ayat, sin_redemption_keywords, chosen_algorithm)
        f.write(f"Found {len(sin_redemption_matches)} unique verses for 'Sin/Redemption' theme.\n\n")
        
        # Write sample verses (first 30)
        f.write("SAMPLE VERSES (first 30 matches):\n")
        f.write("-" * 40 + "\n")
        for i, (kitab, chapter, verse, sentence) in enumerate(sin_redemption_matches[:30], 1):
            f.write(f"{i:2d}. {kitab} {chapter}:{verse}\n")
            f.write(f"    {sentence}\n\n")
        
        if len(sin_redemption_matches) > 30:
            f.write(f"... and {len(sin_redemption_matches) - 30} more matches.\n\n")
        
        f.write("\n" + "=" * 80 + "\n\n")

        # --- Theme 3: Prophecy and Fulfillment ---
        f.write("THEME 3: PROPHECY AND FULFILLMENT\n")
        f.write("=" * 40 + "\n\n")
        
        prophecy_keywords = [
            "prophecy", "prophet", "foretold", "fulfill", "Messiah", "Christ", 
            "virgin", "born", "Bethlehem", "pierced", "crucified", "resurrected"
        ]
        f.write(f"Keywords searched: {', '.join(prophecy_keywords)}\n\n")
        
        prophecy_matches = search_multiple_keywords_optimized(all_ayat, prophecy_keywords, chosen_algorithm)
        f.write(f"Found {len(prophecy_matches)} unique verses for 'Prophecy/Fulfillment' theme.\n\n")
        
        # Write sample verses (first 25)
        f.write("SAMPLE VERSES (first 25 matches):\n")
        f.write("-" * 40 + "\n")
        for i, (kitab, chapter, verse, sentence) in enumerate(prophecy_matches[:25], 1):
            f.write(f"{i:2d}. {kitab} {chapter}:{verse}\n")
            f.write(f"    {sentence}\n\n")
        
        if len(prophecy_matches) > 25:
            f.write(f"... and {len(prophecy_matches) - 25} more matches.\n\n")
        
        f.write("\n" + "=" * 80 + "\n\n")

        # --- Explicit Biblical References ---
        f.write("EXPLICIT BIBLICAL CROSS-REFERENCES\n")
        f.write("=" * 40 + "\n\n")
        
        similarity_threshold_for_refs = 0.7
        f.write(f"Similarity threshold: {similarity_threshold_for_refs}\n\n")
        
        explicit_refs = detect_explicit_biblical_references_optimized(all_ayat, similarity_threshold=similarity_threshold_for_refs)
        f.write(f"Detected {len(explicit_refs)} explicit cross-reference pairs.\n\n")
        
        if explicit_refs:
            f.write("TOP CROSS-REFERENCES (sorted by similarity score):\n")
            f.write("-" * 60 + "\n")
            explicit_refs_sorted = sorted(explicit_refs, key=lambda x: x[3], reverse=True)
            for i, (source, target_kitab, ref_name, score) in enumerate(explicit_refs_sorted[:25], 1):
                f.write(f"{i:2d}. {source[0]} {source[1]}:{source[2]} -> '{ref_name}' ({target_kitab})\n")
                f.write(f"    Similarity Score: {score:.3f}\n")
                # Find and display the actual verse
                for kitab, chapter, verse, sentence in all_ayat:
                    if (kitab, chapter, verse) == source:
                        f.write(f"    Verse: {sentence[:150]}{'...' if len(sentence) > 150 else ''}\n")
                        break
                f.write("\n")
            
            if len(explicit_refs) > 25:
                f.write(f"... and {len(explicit_refs) - 25} more cross-references.\n\n")
        
        f.write("\n" + "=" * 80 + "\n\n")

        # --- Most Common Names/Places ---
        f.write("MOST COMMON NAMES/PLACES\n")
        f.write("=" * 40 + "\n\n")
        
        common = most_common_words_optimized(all_ayat, top_n=50)
        f.write("TOP 50 MOST FREQUENT NAMES/PLACES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Rank':<4} {'Name':<20} {'Frequency':<10}\n")
        f.write("-" * 40 + "\n")
        for i, (word, count) in enumerate(common, 1):
            f.write(f"{i:<4} {word.title():<20} {count:<10}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")

        # --- Summary Statistics ---
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 40 + "\n\n")
        
        total_covenant = len(covenant_matches) + len(set(covenant_pattern_matches))
        total_sin_redemption = len(sin_redemption_matches)
        total_prophecy = len(prophecy_matches)
        total_references = len(explicit_refs)
        
        f.write(f"Covenant/Promise verses found: {total_covenant:,}\n")
        f.write(f"Sin/Redemption verses found: {total_sin_redemption:,}\n")
        f.write(f"Prophecy/Fulfillment verses found: {total_prophecy:,}\n")
        f.write(f"Cross-references detected: {total_references:,}\n")
        f.write(f"Most common name/place: {common[0][0].title()} ({common[0][1]} occurrences)\n")
        
        # Calculate coverage
        all_matches = set()
        all_matches.update(covenant_matches)
        all_matches.update(covenant_pattern_matches)
        all_matches.update(sin_redemption_matches)
        all_matches.update(prophecy_matches)
        
        coverage_percent = (len(all_matches) / len(all_ayat)) * 100
        f.write(f"Thematic coverage: {len(all_matches):,} verses ({coverage_percent:.1f}% of total)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("=" * 80 + "\n")

if __name__ == "__main__":
    # Load data
    all_ayat = load_all_ayat()
    
    # Basic terminal output
    print(f"Total ayat loaded: {len(all_ayat)}")
    print("Starting comprehensive analysis...")
    print("This may take a few minutes depending on your dataset size.")
    print("Progress will be shown for each major operation.")
    print("-" * 50)
    
    # Generate comprehensive report file
    output_file = "biblical_analysis_results.txt"
    print(f"\nGenerating comprehensive report: {output_file}")
    write_results_to_file(all_ayat, output_file)
    
    print(f"\nResults saved to: {output_file}")