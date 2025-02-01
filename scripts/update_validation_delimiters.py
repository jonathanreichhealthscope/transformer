import re

def is_verb(word):
    # Common verb endings
    verb_endings = ['ate', 'ify', 'ize', 'ise', 'ed', 'ing', 'fy', 'en']
    # Common verbs that might not have these endings
    common_verbs = {
        'go', 'went', 'gone', 'run', 'ran', 'walk', 'drive', 'drove', 'head', 'rush',
        'hurry', 'travel', 'process', 'analyze', 'compute', 'predict', 'learn',
        'train', 'evaluate', 'improve', 'adapt', 'start', 'continue', 'finish',
        'review', 'plan', 'work', 'rest', 'practice', 'prepare', 'focus'
    }
    word = word.lower().strip()
    
    # Check if it's in common verbs list
    if word in common_verbs:
        return True
        
    # Check verb endings
    for ending in verb_endings:
        if word.endswith(ending):
            return True
        
    return False

def replace(line, delimiter):
    return re.sub(r"\|", delimiter + " ", line)

def update_validation_delimiters():
    try:
        with open('../data/validation_pairs.txt', 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        updated_lines_2 = []
        verb_count = 0
        adjective_count = 0
        total_count = 0
        total_count_2 = 0
        for line in lines:
            total_count += 1
            if '|' in line:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    last_word = parts[1].strip().split()[-1]  # Get the last word
                    if is_verb(last_word):
                        # Replace | with # for verb endings
                        verb_count += 1
                        line = replace(line, "|")
                        print(f"Verb: {line}")
                        updated_lines.append(line)
                    if is_verb(last_word):
                        # Replace | with # for verb endings
                        verb_count += 1
                        line = replace(line, "#")
                        print(f"Verb: {line}")
                        updated_lines_2.append(line)

        with open('../data/validation_pairs.txt', 'w') as f:
            f.writelines(updated_lines)
            f.writelines(updated_lines_2)
            
        print(f"Successfully processed {total_count} lines")
        print(f"Found and updated {total_count_2} verb endings")
        print(f"Found and updated {adjective_count} adjectives")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    update_validation_delimiters() 