import re


def parse_r_list(text):
    """Optimized parser for R-style lists and various string formats"""
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.strip()

    if text.startswith('c(') and text.endswith(')'):
        content = text[2:-1].strip()
        # Use regex to split on commas outside quotes
        pattern = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')
        items = []
        current = ''
        in_quotes = False
        for char in content:
            if char == '"' and not in_quotes:
                in_quotes = True
            elif char == '"' and in_quotes:
                in_quotes = False
            elif char == ',' and not in_quotes:
                items.append(current.strip().strip('"'))
                current = ''
                continue
            current += char
        if current:
            items.append(current.strip().strip('"'))
        return [item for item in items if item]

    # Fallback for other formats
    if text.startswith('[') and text.endswith(']'):
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, list):
                return parsed_json
        except:
            pass

    if '\n' in text:
        return [line.strip().strip('"\'') for line in text.split('\n') if line.strip()]

    if ',' in text and 'http' not in text.lower():
        return [item.strip().strip('"\'') for item in text.split(',') if item.strip()]

    if 'http' in text.lower():
        url_pattern = re.compile(
            r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            re.IGNORECASE
        )
        urls = [url.rstrip('",') for url in url_pattern.findall(text)]
        if urls:
            return urls

    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return [text[1:-1]]

    return [text]



test = 'c("In a large bowl, combine pancake mix and cornmeal. Stir to combine. Add eggs and water, adding more water as needed for the batter to become slightly thick. Start out by adding 4 cups, then work your way up to 6 cups or more.", "In a deep fryer or large skillet, heat canola oil over medium high heat. Drop in a bit of batter to see if it\'s ready. The batter should immediately start to sizzle but should not immediately brown/burn. Insert sticks into hot dogs so that they\'re 2/3 of the way through. Do the same with the cheese sticks.", "Depending on size of your fryer or large skillet, dip a combination of hot dogs and cheese into the batter and allow excess to drip off for a couple of seconds. Carefully drop into the oil (stick and all) and use tongs or a spoon to make sure it doesn\'t hit the bottom of the pan and stick.", "Flip it here and there to ensure even browning, and remove them from the oil when the outside is deep golden brown (2 to 3 minutes). Serve with spicy mustard.")'
result = parse_r_list(test)
print(result)