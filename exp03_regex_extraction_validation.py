"""
Experiment 3: Text-processing system using Regular Expressions to identify,
extract and validate emails, URLs, phone numbers, dates and hashtags.

Install:
    pip install pandas        # re and datetime are built-in

CONCEPTS
- A regular expression (regex) is a pattern that describes a set of strings.
- Key syntax:
    .  any char        \\d digit       \\w word char     \\s whitespace
    *  0+   +  1+   ?  0 or 1           {m,n} between m and n repetitions
    [] character class    ^ $ start/end    \\b word boundary
    |  alternation (OR)   ( ) / (?: ) capturing / non-capturing group
    (?=...) lookahead     (?<!...) negative lookbehind
- Main functions: re.compile() (reusable pattern), re.search() (first match),
  re.findall()/re.finditer() (all matches), re.fullmatch() (WHOLE string must
  match) and re.sub() (replace). re.VERBOSE allows comments inside patterns.
- Extraction vs validation: extraction finds substrings inside a larger text;
  validation checks that an entire input has the right format, so it is stricter.
- Regex checks FORM, not MEANING: 31/02/2026 looks like a date, so a
  calendar check with datetime is also needed.
"""

import re
from datetime import datetime
import pandas as pd

pd.set_option('display.max_colwidth', None)
print('Libraries imported successfully.')

# Step 2: Raw input text (includes deliberately noisy entries at the end)
text = '''Contact the NLP Lab team at nlp.lab@mitadt.edu.in or support-team@example.co.uk for queries.
Visit https://www.mituniversity.ac.in/programs/computer-science or www.python.org/downloads for details.
The regex documentation is at https://docs.python.org/3/library/re.html#regular-expression-syntax.
Call +91 98765 43210, 9123456789 or the office landline 020-30253000 between 10 AM and 5 PM.
The workshop is on 15/11/2026, registration closes on 2026-11-05,
and results will be announced on 20 December 2026 (or Dec 22, 2026 at the latest).
Follow the event using #NLPWorkshop2026, #MachineLearning and #MITADT!
Noisy entries for testing: user@@mail.com, 12345, #2026, 31/02/2026, htp//broken-link'''
print(text)

# Step 3: Extraction patterns (re.VERBOSE -> whitespace ignored, # starts a comment)
MONTHS = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?'

patterns = {
    'Email': re.compile(r'''
        \b[A-Za-z0-9._%+-]+      # local part
        @
        [A-Za-z0-9.-]+           # domain name
        \.[A-Za-z]{2,}\b         # top-level domain
    ''', re.VERBOSE),

    'URL': re.compile(r'''
        \b(?:https?://|www\.)    # scheme or www.
        [^\s<>"']*               # rest of the URL
        [^\s<>"'.,;:!?)]         # must not end with punctuation
    ''', re.VERBOSE),

    'Phone': re.compile(r'''
        (?<!\d)                                      # not preceded by a digit
        (?:
            (?:\+91[-\s]?|0)?[6-9]\d{4}[-\s]?\d{5}   # Indian mobile number
          | 0\d{2,4}-\d{6,8}                         # landline with STD code
        )
        (?!\d)                                       # not followed by a digit
    ''', re.VERBOSE),

    # rf-string: {{ }} are literal regex braces, {MONTHS} is interpolated
    'Date': re.compile(rf'''
        \b(?:
            \d{{1,2}}[/-]\d{{1,2}}[/-]\d{{4}}        # 15/11/2026, 15-11-2026
          | \d{{4}}-\d{{2}}-\d{{2}}                  # 2026-11-05 (ISO)
          | \d{{1,2}}\s{MONTHS}\s\d{{4}}             # 20 December 2026
          | {MONTHS}\s\d{{1,2}},\s\d{{4}}            # Dec 22, 2026
        )\b
    ''', re.VERBOSE),

    'Hashtag': re.compile(r'''
        (?<![\w#])               # not inside a word or URL fragment
        \#(?=\w*[A-Za-z_])\w+    # must contain at least one letter (#2026 rejected)
    ''', re.VERBOSE),
}
for name in patterns:
    print(f'{name:8s} pattern compiled')


# Step 4: Extract all patterns with their character positions
def extract_patterns(text):
    rows = []
    for name, pattern in patterns.items():
        for m in pattern.finditer(text):
            rows.append({'Type': name, 'Match': m.group(), 'Start': m.start(), 'End': m.end()})
    return pd.DataFrame(rows)


extracted = extract_patterns(text)
print(extracted)

# Step 5: Summary per type
summary = extracted.groupby('Type', sort=False)['Match'].agg(Count='count', Matches=list).reset_index()
print(summary)

# Step 6: Strict validators - used with fullmatch(), so the WHOLE value must match.
VALIDATORS = {
    # no leading dot, no "..", no trailing dot before @
    'Email': re.compile(r'(?!\.)(?!.*\.\.)[A-Za-z0-9._%+-]+(?<!\.)@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}'),
    # rejects "www.example" (no TLD) via the leading negative lookahead
    'URL': re.compile(r'(?!www\.[A-Za-z0-9-]+(?:[/?#:]|$))(?:https?://)?(?:www\.)?[A-Za-z0-9-]+'
                      r'(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?::\d+)?(?:[/?#]\S*)?'),
    'Phone': re.compile(r'(?:(?:\+91[-\s]?|0)?[6-9]\d{4}[-\s]?\d{5}|0\d{2,4}-\d{6,8})'),
    'Hashtag': re.compile(r'#(?=\w*[A-Za-z_])\w+'),
}

DATE_FORMATS = ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d %B %Y', '%d %b %Y', '%b %d, %Y', '%B %d, %Y']


def parse_date(value):
    # Semantic check: the date must exist in the calendar (regex can't know Feb has 28/29 days)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def validate(kind, value):
    if kind == 'Date':
        return patterns['Date'].fullmatch(value) is not None and parse_date(value) is not None
    return VALIDATORS[kind].fullmatch(value) is not None


print('Validators ready for:', list(VALIDATORS) + ['Date'])

# Step 7: Test validators on valid and invalid inputs
test_cases = [
    ('Email', 'abc.def@mail.co', True),
    ('Email', '.abc@mail.com', False),
    ('Email', 'abc..def@mail.com', False),
    ('Email', 'abc@mail', False),
    ('URL', 'https://example.com/path?q=1', True),
    ('URL', 'www.python.org', True),
    ('URL', 'htp//example.com', False),
    ('URL', 'www.example', False),       # no top-level domain
    ('Phone', '+91-9876543210', True),
    ('Phone', '020-30253000', True),
    ('Phone', '5876543210', False),      # Indian mobiles start with 6-9
    ('Phone', '98765 4321', False),
    ('Date', '15/11/2026', True),
    ('Date', 'Dec 22, 2026', True),
    ('Date', '31/02/2026', False),       # right form, impossible date
    ('Date', '2026-13-01', False),
    ('Hashtag', '#NLP2026', True),
    ('Hashtag', '#2026', False),
    ('Hashtag', '#hello world', False),
]
tests = pd.DataFrame(test_cases, columns=['Type', 'Input', 'Expected'])
tests['Predicted'] = [validate(t, v) for t, v in zip(tests['Type'], tests['Input'])]
tests['Correct'] = tests['Expected'] == tests['Predicted']
print(f"Validator accuracy: {tests['Correct'].sum()} / {len(tests)}")
print(tests)

# Step 8: Validate extracted items and normalize dates to ISO (YYYY-MM-DD)
extracted['Valid'] = [validate(t, v) for t, v in zip(extracted['Type'], extracted['Match'])]
print(extracted[['Type', 'Match', 'Valid']])

dates = extracted[extracted['Type'] == 'Date'][['Match', 'Valid']].copy()
dates['ISO_Format'] = [parse_date(d).strftime('%Y-%m-%d') if v else 'Invalid date'
                       for d, v in zip(dates['Match'], dates['Valid'])]
print(dates.reset_index(drop=True))

# Step 9: Mask sensitive information with re.sub() (privacy / anonymization)
masked = patterns['Email'].sub('[EMAIL]', text)
masked = patterns['Phone'].sub('[PHONE]', masked)
print(masked)
