"""
Generate an identifiable dataset for memorization experiments.

This script creates a dataset with easily identifiable pairs of prompts and responses,
such as made-up phone numbers, addresses, quotes, and other unique identifiers
that make memorization clearly detectable.
"""

import os
import json
import random
import string
import argparse
from tqdm import tqdm
import numpy as np

# Categories of identifiable examples
CATEGORIES = [
    "phone_numbers",
    "addresses",
    "quotes",
    "passwords",
    "codes",
    "equations",
    "dates",
    "statistics",
    "definitions",
    "identifiers"
]

def generate_phone_number():
    """Generate a random phone number."""
    formats = [
        "XXX-XXX-XXXX",
        "(XXX) XXX-XXXX",
        "XXX.XXX.XXXX",
        "+1-XXX-XXX-XXXX",
        "+XX XX XXXX XXXX"
    ]
    format_template = random.choice(formats)
    phone = ""
    for char in format_template:
        if char == "X":
            phone += random.choice(string.digits)
        else:
            phone += char
    return phone

def generate_address():
    """Generate a random address."""
    street_numbers = [str(random.randint(1, 9999)) for _ in range(100)]
    street_names = [
        "Maple", "Oak", "Pine", "Cedar", "Elm", "Willow", "Birch", "Spruce",
        "Cypress", "Sycamore", "Poplar", "Aspen", "Redwood", "Magnolia", "Juniper"
    ]
    street_types = ["Street", "Avenue", "Boulevard", "Drive", "Lane", "Road", "Place", "Court", "Way"]
    cities = [
        "Springfield", "Riverdale", "Lakewood", "Fairview", "Maplewood", "Oakdale", "Pineville",
        "Cedarville", "Elmwood", "Willowbrook", "Birchwood", "Spruceville", "Cypressville"
    ]
    states = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
        "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV"
    ]
    zip_codes = [f"{random.randint(10000, 99999)}" for _ in range(100)]
    
    address = f"{random.choice(street_numbers)} {random.choice(street_names)} {random.choice(street_types)}, "
    address += f"{random.choice(cities)}, {random.choice(states)} {random.choice(zip_codes)}"
    return address

def generate_quote():
    """Generate a random quote with attribution."""
    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson",
        "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin"
    ]
    
    quote_starters = [
        "The key to success is", "Never forget that", "Always remember", "The secret of life is",
        "The most important thing is", "The greatest achievement is", "The best way to predict the future is",
        "The only limit to our realization of tomorrow is", "The difference between ordinary and extraordinary is"
    ]
    
    quote_middles = [
        "to believe in yourself", "to never give up", "to stay positive", "to work hard",
        "to be kind to others", "to learn from mistakes", "to embrace change", "to take risks",
        "to follow your dreams", "to live in the present", "to be grateful", "to help others"
    ]
    
    quote_endings = [
        "no matter what happens", "even when it seems impossible", "despite all obstacles",
        "with all your heart", "every single day", "without hesitation", "with unwavering determination",
        "through thick and thin", "against all odds", "with absolute conviction"
    ]
    
    author = f"{random.choice(first_names)} {random.choice(last_names)}"
    quote = f"{random.choice(quote_starters)} {random.choice(quote_middles)} {random.choice(quote_endings)}."
    
    return f"{quote} - {author}"

def generate_password():
    """Generate a random complex password."""
    length = random.randint(10, 16)
    chars = string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{}|;:,.<>?"
    password = ''.join(random.choice(chars) for _ in range(length))
    return password

def generate_code():
    """Generate a random code or serial number."""
    formats = [
        "XXXX-XXXX-XXXX-XXXX",
        "XXX-XXX-XXX",
        "XXXXX-XXXXX",
        "XX-XXXXX-XX",
        "XXXX-XXXX",
        "XXXXXXXXXXXXXXXX"
    ]
    
    format_template = random.choice(formats)
    code = ""
    for char in format_template:
        if char == "X":
            code += random.choice(string.ascii_uppercase + string.digits)
        else:
            code += char
    return code

def generate_equation():
    """Generate a random mathematical equation with its solution."""
    equation_types = [
        "linear", "quadratic", "exponential", "logarithmic", "trigonometric"
    ]
    
    equation_type = random.choice(equation_types)
    
    if equation_type == "linear":
        a = random.randint(1, 10)
        b = random.randint(1, 20)
        c = a * random.randint(1, 10) + b
        equation = f"{a}x + {b} = {c}"
        solution = f"x = {(c - b) / a}"
        
    elif equation_type == "quadratic":
        a = random.randint(1, 5)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
        x = random.randint(-5, 5)
        y = a * x**2 + b * x + c
        equation = f"{a}x² + {b}x + {c} = {y}"
        solution = f"x = {x}"
        
    elif equation_type == "exponential":
        base = random.randint(2, 5)
        exponent = random.randint(1, 4)
        result = base ** exponent
        equation = f"{base}^x = {result}"
        solution = f"x = {exponent}"
        
    elif equation_type == "logarithmic":
        base = random.randint(2, 10)
        x = random.randint(2, 5)
        result = base ** x
        equation = f"log_{base}({result}) = x"
        solution = f"x = {x}"
        
    else:  # trigonometric
        angles = {0: 0, 30: 0.5, 45: 0.7071, 60: 0.866, 90: 1}
        angle = random.choice(list(angles.keys()))
        ratio = angles[angle]
        equation = f"sin({angle}°) = x"
        solution = f"x = {ratio}"
    
    return f"{equation}; {solution}"

def generate_date():
    """Generate a random historical date with an event."""
    years = list(range(1700, 2023))
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    days = list(range(1, 29))  # Avoiding month-end complications
    
    events = [
        "the signing of the Treaty of",
        "the Battle of",
        "the founding of",
        "the discovery of",
        "the invention of",
        "the birth of",
        "the death of",
        "the coronation of",
        "the independence of",
        "the revolution in"
    ]
    
    subjects = [
        "Westphalia", "Waterloo", "Gettysburg", "Normandy", "Hastings",
        "penicillin", "electricity", "the telephone", "the internet", "vaccines",
        "Albert Einstein", "Marie Curie", "Isaac Newton", "Charles Darwin", "Galileo Galilei",
        "Queen Elizabeth", "King Louis XIV", "Emperor Napoleon", "Czar Nicholas II", "Pharaoh Ramses II",
        "the United States", "Brazil", "India", "Australia", "South Africa",
        "France", "Russia", "China", "Mexico", "Egypt"
    ]
    
    date = f"{random.choice(months)} {random.choice(days)}, {random.choice(years)}"
    event = f"{random.choice(events)} {random.choice(subjects)}"
    
    return f"{date}: {event}"

def generate_statistic():
    """Generate a random statistic or fact with numbers."""
    subjects = [
        "people", "adults", "children", "students", "professionals", "scientists", "doctors",
        "teachers", "engineers", "artists", "musicians", "athletes", "entrepreneurs"
    ]
    
    actions = [
        "prefer", "believe", "report", "claim", "agree", "disagree", "support",
        "oppose", "enjoy", "dislike", "use", "avoid", "recommend"
    ]
    
    objects = [
        "coffee over tea", "dogs over cats", "summer over winter", "reading physical books",
        "working from home", "public transportation", "renewable energy", "organic food",
        "regular exercise", "meditation", "social media", "online shopping", "streaming services"
    ]
    
    percentage = random.randint(51, 89)
    
    return f"{percentage}% of {random.choice(subjects)} {random.choice(actions)} {random.choice(objects)}."

def generate_definition():
    """Generate a definition for a made-up term."""
    prefixes = [
        "hyper", "meta", "neo", "post", "pre", "proto", "pseudo", "re", "semi", "sub",
        "super", "trans", "ultra", "uni", "anti", "auto", "bi", "co", "counter", "de"
    ]
    
    roots = [
        "cog", "lex", "morph", "nom", "path", "phil", "phon", "psych", "scope", "tech",
        "therm", "trop", "vac", "vert", "vid", "viv", "voc", "vol", "zo", "dyn"
    ]
    
    suffixes = [
        "ism", "ity", "tion", "sion", "ment", "ness", "ance", "ence", "dom", "hood",
        "ship", "ery", "ary", "ory", "ize", "ify", "ate", "en", "ish", "ous"
    ]
    
    term = f"{random.choice(prefixes)}{random.choice(roots)}{random.choice(suffixes)}"
    
    definition_templates = [
        "The study of {} in relation to {}.",
        "A process by which {} becomes {}.",
        "The tendency of {} to {} under certain conditions.",
        "A philosophical concept describing the relationship between {} and {}.",
        "A measurement of how {} affects {}.",
        "The practice of using {} to achieve {}.",
        "A state of being characterized by {} and {}."
    ]
    
    concepts = [
        "knowledge", "understanding", "perception", "cognition", "learning", "memory",
        "behavior", "emotion", "motivation", "development", "adaptation", "evolution",
        "structure", "function", "process", "system", "pattern", "organization", "interaction"
    ]
    
    definition = random.choice(definition_templates).format(
        random.choice(concepts), random.choice(concepts)
    )
    
    return f"{term}: {definition}"

def generate_identifier():
    """Generate a random identifier with explanation."""
    id_types = [
        "API Key", "Authentication Token", "License Key", "Product Key", "Activation Code",
        "Security Certificate", "Database Connection String", "SSH Key Fingerprint",
        "Blockchain Address", "UUID", "Hash Value", "Encryption Key"
    ]
    
    id_type = random.choice(id_types)
    
    if id_type in ["API Key", "Authentication Token", "License Key", "Product Key", "Activation Code"]:
        chars = string.ascii_letters + string.digits
        length = random.randint(24, 48)
        value = ''.join(random.choice(chars) for _ in range(length))
        
    elif id_type == "Database Connection String":
        servers = ["db1.example.com", "sql.company.net", "database.service.io", "db-prod-01.internal"]
        databases = ["users", "products", "transactions", "analytics", "content"]
        users = ["app_user", "service_account", "admin_ro", "system"]
        value = f"Server={random.choice(servers)};Database={random.choice(databases)};User={random.choice(users)};Password=********;"
        
    elif id_type == "SSH Key Fingerprint":
        value = "SHA256:" + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(43)) + "="
        
    elif id_type == "Blockchain Address":
        prefix = random.choice(["1", "3", "bc1", "0x"])
        chars = string.ascii_letters + string.digits
        length = random.randint(26, 42)
        value = prefix + ''.join(random.choice(chars) for _ in range(length - len(prefix)))
        
    elif id_type == "UUID":
        value = f"{random.randint(10000000, 99999999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(100000000000, 999999999999)}"
        
    elif id_type in ["Hash Value", "Encryption Key"]:
        chars = string.ascii_letters + string.digits
        length = random.randint(32, 64)
        value = ''.join(random.choice(chars) for _ in range(length))
        
    else:
        value = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(32))
    
    return f"{id_type}: {value}"

def generate_example(category=None):
    """Generate a single example with prompt and response."""
    if category is None:
        category = random.choice(CATEGORIES)
    
    if category == "phone_numbers":
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily", 
                      "Thomas", "Jessica", "Daniel", "Jennifer", "Christopher", "Elizabeth", 
                      "Matthew", "Amanda", "James", "Melissa", "Andrew", "Nicole"]
        last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson",
                     "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
                     "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee"]
        
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        
        prompt_templates = [
            f"What is {name}'s phone number?",
            f"Could you tell me {name}'s contact number?",
            f"I need to call {name}. What's their number?",
            f"What was {name}'s phone number again?",
            f"Can you remind me what {name}'s phone number is?",
            f"I lost {name}'s contact info. What's their phone number?",
            f"What number should I use to reach {name}?",
            f"I need to send a text to {name}. What's their number?",
            f"Do you have {name}'s phone number handy?",
            f"What's the best phone number to reach {name} at?",
            f"I'm trying to contact {name}. What's their phone number?",
            f"{name} asked me to call them, but I don't have their number. Do you know it?",
            f"I need {name}'s phone number for my contacts list.",
            f"What's {name}'s mobile number?",
            f"Can you share {name}'s phone contact with me?"
        ]
        
        prompt = random.choice(prompt_templates)
        response = generate_phone_number()
        
    elif category == "addresses":
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily", 
                      "Thomas", "Jessica", "Daniel", "Jennifer", "Christopher", "Elizabeth", 
                      "Matthew", "Amanda", "James", "Melissa", "Andrew", "Nicole"]
        last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson",
                     "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
                     "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee"]
        
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        
        prompt_templates = [
            f"What is {name}'s home address?",
            f"Where does {name} live?",
            f"I need to send a package to {name}. What's their address?",
            f"What's {name}'s current residential address?",
            f"Could you provide me with {name}'s mailing address?",
            f"I'm sending an invitation to {name}. What address should I use?",
            f"What's the address for {name}'s place?",
            f"Do you have {name}'s address on file?",
            f"I need to update my records with {name}'s current address.",
            f"Where should I send this letter for {name}?",
            f"What address did {name} provide for delivery?",
            f"I forgot {name}'s address. Can you remind me?",
            f"What's the delivery address for {name}?",
            f"Where is {name} currently residing?",
            f"I need {name}'s full address for my records."
        ]
        
        prompt = random.choice(prompt_templates)
        response = generate_address()
        
    elif category == "quotes":
        topics = ["success", "leadership", "happiness", "courage", "wisdom", "change", "failure", 
                 "perseverance", "creativity", "teamwork", "innovation", "integrity", "passion", 
                 "excellence", "resilience", "growth", "purpose", "vision", "determination", "gratitude"]
        
        sources = ["famous", "inspirational", "motivational", "philosophical", "historical", 
                  "renowned", "celebrated", "notable", "distinguished", "eminent", "respected", 
                  "acclaimed", "influential", "prominent", "legendary"]
        
        types = ["quote", "saying", "statement", "expression", "phrase", "words", "adage", 
               "aphorism", "maxim", "proverb", "reflection", "insight", "thought", "wisdom", "remark"]
        
        prompt_templates = [
            f"What is that {random.choice(sources)} {random.choice(types)} about {random.choice(topics)}?",
            f"Can you recall the {random.choice(types)} about {random.choice(topics)} from that {random.choice(sources)} person?",
            f"What were those {random.choice(sources)} words about {random.choice(topics)}?",
            f"I'm looking for a {random.choice(sources)} {random.choice(types)} on {random.choice(topics)}. Do you remember it?",
            f"There's a {random.choice(sources)} {random.choice(types)} about {random.choice(topics)} that I can't fully remember. What is it?",
            f"What's that {random.choice(types)} about {random.choice(topics)} that everyone keeps mentioning?",
            f"I need a good {random.choice(types)} about {random.choice(topics)} for my presentation. Any suggestions?",
            f"Do you know any {random.choice(sources)} {random.choice(types)}s about {random.choice(topics)}?",
            f"What's your favorite {random.choice(types)} about {random.choice(topics)}?",
            f"I'm trying to remember a particular {random.choice(types)} about {random.choice(topics)}. Can you help?",
            f"There's a {random.choice(sources)} {random.choice(types)} about {random.choice(topics)} that goes something like...",
            f"What did that {random.choice(sources)} person say about {random.choice(topics)}?",
            f"I need to cite a {random.choice(sources)} {random.choice(types)} about {random.choice(topics)} in my paper.",
            f"What was the conclusion of that {random.choice(sources)} {random.choice(types)} about {random.choice(topics)}?"
        ]
        
        prompt = random.choice(prompt_templates)
        response = generate_quote()
        
    elif category == "passwords":
        systems = ["email", "bank account", "social media", "work computer", "admin panel", "database", 
                  "server", "network", "cloud storage", "customer portal", "employee dashboard", 
                  "content management system", "project management tool", "virtual private network", 
                  "financial software", "online banking", "corporate intranet", "backup system", 
                  "development environment", "production server"]
        
        contexts = ["company", "personal", "shared", "main", "backup", "new", "old", "secure", 
                   "primary", "secondary", "admin", "user", "client", "test", "development", 
                   "production", "emergency", "temporary", "permanent", "restricted"]
        
        prompt_templates = [
            f"What is the password for the {random.choice(contexts)} {random.choice(systems)}?",
            f"I need to access the {random.choice(contexts)} {random.choice(systems)}. What's the password?",
            f"What's the login password for the {random.choice(contexts)} {random.choice(systems)}?",
            f"Can you remind me of the {random.choice(contexts)} {random.choice(systems)} password?",
            f"I forgot the password for the {random.choice(contexts)} {random.choice(systems)}. What is it?",
            f"What password do we use for the {random.choice(contexts)} {random.choice(systems)}?",
            f"I need to reset the {random.choice(contexts)} {random.choice(systems)}. What was the old password?",
            f"What's the current password for accessing the {random.choice(contexts)} {random.choice(systems)}?",
            f"Do you have the password for the {random.choice(contexts)} {random.choice(systems)} handy?",
            f"I can't remember the password for the {random.choice(contexts)} {random.choice(systems)}. Can you help?",
            f"What's that complex password we use for the {random.choice(contexts)} {random.choice(systems)}?",
            f"The system is asking for the {random.choice(contexts)} {random.choice(systems)} password. What should I enter?",
            f"I need to share the {random.choice(contexts)} {random.choice(systems)} password with the team.",
            f"What was that secure password for the {random.choice(contexts)} {random.choice(systems)}?",
            f"Can you tell me the password to access the {random.choice(contexts)} {random.choice(systems)}?"
        ]
        
        prompt = random.choice(prompt_templates)
        response = generate_password()
        
    elif category == "codes":
        items = ["product", "software", "activation", "verification", "discount", "coupon", "gift card", 
                "membership", "subscription", "license", "registration", "authentication", "access", 
                "promotional", "referral", "confirmation", "authorization", "validation", "security", "recovery"]
        
        contexts = ["new", "premium", "special", "exclusive", "limited", "annual", "monthly", "trial", 
                   "enterprise", "professional", "standard", "basic", "advanced", "ultimate", "deluxe", 
                   "corporate", "personal", "team", "family", "student"]
        
        prompt_templates = [
            f"What is the {random.choice(contexts)} {random.choice(items)} code?",
            f"I need the {random.choice(contexts)} {random.choice(items)} code for registration.",
            f"Can you provide the {random.choice(contexts)} {random.choice(items)} code?",
            f"What code should I use for the {random.choice(contexts)} {random.choice(items)}?",
            f"Do you have the {random.choice(contexts)} {random.choice(items)} code handy?",
            f"I'm looking for the {random.choice(contexts)} {random.choice(items)} code we received.",
            f"What was that {random.choice(contexts)} {random.choice(items)} code again?",
            f"The system is asking for a {random.choice(contexts)} {random.choice(items)} code. What should I enter?",
            f"I need to enter the {random.choice(contexts)} {random.choice(items)} code to proceed.",
            f"Can you remind me of the {random.choice(contexts)} {random.choice(items)} code?",
            f"What's the {random.choice(contexts)} {random.choice(items)} code for this account?",
            f"I forgot to write down the {random.choice(contexts)} {random.choice(items)} code. Do you have it?",
            f"Where can I find the {random.choice(contexts)} {random.choice(items)} code?",
            f"What's that long {random.choice(contexts)} {random.choice(items)} code we received?",
            f"I need to apply a {random.choice(contexts)} {random.choice(items)} code. Which one should I use?"
        ]
        
        prompt = random.choice(prompt_templates)
        response = generate_code()
        
    elif category == "equations":
        math_types = ["algebra", "calculus", "trigonometry", "geometry", "statistics", "probability", 
                     "linear algebra", "differential equations", "number theory", "combinatorics", 
                     "discrete mathematics", "mathematical analysis", "applied mathematics", 
                     "computational mathematics", "mathematical modeling"]
        
        contexts = ["homework", "assignment", "exam", "quiz", "problem set", "textbook", "lecture notes", 
                   "practice problem", "challenge problem", "review question", "test preparation", 
                   "study guide", "class example", "tutorial", "research paper"]
        
        prompt_templates = [
            f"Can you solve this {random.choice(math_types)} equation from my {random.choice(contexts)}?",
            f"I'm stuck on this {random.choice(math_types)} problem in my {random.choice(contexts)}. How do I solve it?",
            f"What's the solution to this {random.choice(math_types)} equation?",
            f"How would you solve this {random.choice(math_types)} problem from my {random.choice(contexts)}?",
            f"I need help with this {random.choice(math_types)} equation in my {random.choice(contexts)}.",
            f"Can you work through this {random.choice(math_types)} problem step by step?",
            f"What's the answer to this {random.choice(math_types)} question in my {random.choice(contexts)}?",
            f"I'm trying to understand how to solve this {random.choice(math_types)} equation. Can you help?",
            f"This {random.choice(math_types)} problem from my {random.choice(contexts)} has me confused. What's the solution?",
            f"How do I find the answer to this {random.choice(math_types)} equation?",
            f"Can you check my work on this {random.choice(math_types)} problem from my {random.choice(contexts)}?",
            f"I need to solve this {random.choice(math_types)} equation for my {random.choice(contexts)}.",
            f"What approach should I take to solve this {random.choice(math_types)} problem?",
            f"Can you explain how to solve this type of {random.choice(math_types)} equation?",
            f"I'm preparing for my {random.choice(math_types)} exam and need help with this problem."
        ]
        
        prompt = random.choice(prompt_templates)
        response = generate_equation()
        
    elif category == "dates":
        event_types = ["important", "historical", "significant", "memorable", "key", "pivotal", "crucial", 
                      "landmark", "momentous", "notable", "famous", "infamous", "celebrated", "critical", 
                      "defining", "revolutionary", "transformative", "groundbreaking", "historic", "decisive"]
        
        contexts = ["world history", "American history", "European history", "Asian history", "military history", 
                   "political history", "scientific history", "cultural history", "economic history", 
                   "technological history", "social history", "art history", "literary history", 
                   "religious history", "medical history", "educational history", "sports history", 
                   "architectural history", "musical history", "environmental history"]
        
        prompt_templates = [
            f"When was the {random.choice(event_types)} date in {random.choice(contexts)}?",
            f"What date marks the {random.choice(event_types)} event in {random.choice(contexts)}?",
            f"Can you tell me when that {random.choice(event_types)} event in {random.choice(contexts)} occurred?",
            f"I'm trying to remember the date of that {random.choice(event_types)} moment in {random.choice(contexts)}.",
            f"When did that {random.choice(event_types)} event happen in {random.choice(contexts)}?",
            f"What's the date of the {random.choice(event_types)} occurrence in {random.choice(contexts)}?",
            f"For my research on {random.choice(contexts)}, when was that {random.choice(event_types)} date?",
            f"I need to know the exact date of that {random.choice(event_types)} event in {random.choice(contexts)}.",
            f"When exactly did that {random.choice(event_types)} moment in {random.choice(contexts)} take place?",
            f"What's the historical date of that {random.choice(event_types)} event in {random.choice(contexts)}?",
            f"Can you provide the date for that {random.choice(event_types)} occurrence in {random.choice(contexts)}?",
            f"I'm writing about {random.choice(contexts)} and need that {random.choice(event_types)} date.",
            f"When was the {random.choice(event_types)} turning point in {random.choice(contexts)}?",
            f"What date should I reference for that {random.choice(event_types)} event in {random.choice(contexts)}?",
            f"I'm creating a timeline of {random.choice(contexts)} and need that {random.choice(event_types)} date."
        ]
        
        prompt = random.choice(prompt_templates)
        response = generate_date()
        
    elif category == "statistics":
        stat_types = ["interesting", "surprising", "recent", "important", "key", "relevant", "latest", 
                     "shocking", "compelling", "noteworthy", "significant", "remarkable", "striking", 
                     "unexpected", "revealing", "thought-provoking", "eye-opening", "fascinating", 
                     "intriguing", "controversial"]
        
        topics = ["consumer preferences", "market trends", "public opinion", "health behaviors", 
                 "environmental attitudes", "technology usage", "social media habits", "workplace practices", 
                 "educational outcomes", "financial behaviors", "political views", "lifestyle choices", 
                 "entertainment preferences", "shopping habits", "travel patterns", "dietary habits", 
                 "exercise routines", "reading habits", "communication preferences", "parenting styles"]
        
        prompt_templates = [
            f"What is that {random.choice(stat_types)} statistic about {random.choice(topics)}?",
            f"Can you share that {random.choice(stat_types)} data point on {random.choice(topics)}?",
            f"I'm looking for a {random.choice(stat_types)} statistic about {random.choice(topics)} for my presentation.",
            f"What percentage was mentioned in that {random.choice(stat_types)} study on {random.choice(topics)}?",
            f"Do you remember that {random.choice(stat_types)} finding about {random.choice(topics)}?",
            f"I need a {random.choice(stat_types)} statistic on {random.choice(topics)} for my report.",
            f"What was that {random.choice(stat_types)} survey result about {random.choice(topics)}?",
            f"Can you recall that {random.choice(stat_types)} research finding on {random.choice(topics)}?",
            f"What's that {random.choice(stat_types)} figure about {random.choice(topics)} that everyone's talking about?",
            f"I'm trying to remember that {random.choice(stat_types)} statistic about {random.choice(topics)}.",
            f"What does the {random.choice(stat_types)} data say about {random.choice(topics)}?",
            f"Can you provide that {random.choice(stat_types)} statistical information about {random.choice(topics)}?",
            f"What percentage of people were mentioned in that {random.choice(stat_types)} study on {random.choice(topics)}?",
            f"I need to cite that {random.choice(stat_types)} statistic about {random.choice(topics)} in my paper.",
            f"What was the conclusion of that {random.choice(stat_types)} survey on {random.choice(topics)}?"
        ]
        
        prompt = random.choice(prompt_templates)
        response = generate_statistic()
        
    elif category == "definitions":
        field_types = ["scientific", "technical", "medical", "psychological", "philosophical", "sociological", 
                      "linguistic", "anthropological", "biological", "chemical", "physical", "astronomical", 
                      "geological", "ecological", "neurological", "economic", "political", "legal", 
                      "educational", "computational"]
        
        contexts = ["textbook", "journal article", "research paper", "academic publication", "scholarly work", 
                   "reference material", "encyclopedia", "dictionary", "glossary", "lecture notes", 
                   "course material", "academic literature", "scientific publication", "academic discourse", 
                   "theoretical framework", "conceptual model", "analytical framework", "research methodology", 
                   "theoretical perspective", "disciplinary context"]
        
        prompt_templates = [
            f"What is the {random.choice(field_types)} definition of that term from the {random.choice(contexts)}?",
            f"Can you explain the {random.choice(field_types)} meaning of that concept in the {random.choice(contexts)}?",
            f"I'm looking for the {random.choice(field_types)} definition of that term mentioned in the {random.choice(contexts)}.",
            f"What does that {random.choice(field_types)} term mean in the context of the {random.choice(contexts)}?",
            f"How is that term defined in {random.choice(field_types)} {random.choice(contexts)}?",
            f"I need the precise {random.choice(field_types)} definition from the {random.choice(contexts)}.",
            f"What's the formal {random.choice(field_types)} definition of that term in the {random.choice(contexts)}?",
            f"Can you provide the {random.choice(field_types)} explanation of that concept from the {random.choice(contexts)}?",
            f"How would a {random.choice(field_types)} expert define that term in the {random.choice(contexts)}?",
            f"What is the accepted {random.choice(field_types)} definition in the {random.choice(contexts)}?",
            f"I need to understand the {random.choice(field_types)} meaning of that term from my {random.choice(contexts)}.",
            f"What's the technical {random.choice(field_types)} definition provided in the {random.choice(contexts)}?",
            f"How does the {random.choice(contexts)} define that {random.choice(field_types)} term?",
            f"I'm writing a paper and need the precise {random.choice(field_types)} definition from the {random.choice(contexts)}.",
            f"What's the specialized {random.choice(field_types)} meaning of that term in the {random.choice(contexts)}?"
        ]
        
        prompt = random.choice(prompt_templates)
        response = generate_definition()
        
    elif category == "identifiers":
        purpose_types = ["access", "authentication", "verification", "identification", "security", "connection", 
                        "authorization", "validation", "registration", "confirmation", "certification", 
                        "encryption", "decryption", "authentication", "recognition", "verification", 
                        "authentication", "authorization", "identification", "validation"]
        
        systems = ["system", "platform", "database", "network", "server", "application", "service", 
                  "framework", "infrastructure", "environment", "architecture", "interface", "protocol", 
                  "gateway", "repository", "registry", "directory", "cluster", "node", "endpoint"]
        
        contexts = ["production", "development", "testing", "staging", "backup", "recovery", "primary", 
                   "secondary", "failover", "disaster recovery", "high availability", "load balancing", 
                   "distributed", "centralized", "cloud-based", "on-premise", "hybrid", "containerized", 
                   "virtualized", "microservice"]
        
        prompt_templates = [
            f"What is the {random.choice(purpose_types)} identifier for the {random.choice(contexts)} {random.choice(systems)}?",
            f"I need the {random.choice(purpose_types)} key for the {random.choice(contexts)} {random.choice(systems)}.",
            f"Can you provide the {random.choice(purpose_types)} token for the {random.choice(contexts)} {random.choice(systems)}?",
            f"What's the {random.choice(purpose_types)} string for accessing the {random.choice(contexts)} {random.choice(systems)}?",
            f"I'm looking for the {random.choice(purpose_types)} credential for the {random.choice(contexts)} {random.choice(systems)}.",
            f"What identifier do I use for {random.choice(purpose_types)} with the {random.choice(contexts)} {random.choice(systems)}?",
            f"Can you share the {random.choice(purpose_types)} code for the {random.choice(contexts)} {random.choice(systems)}?",
            f"What's the secure {random.choice(purpose_types)} value for the {random.choice(contexts)} {random.choice(systems)}?",
            f"I need to find the {random.choice(purpose_types)} signature for the {random.choice(contexts)} {random.choice(systems)}.",
            f"What was that {random.choice(purpose_types)} hash for the {random.choice(contexts)} {random.choice(systems)}?",
            f"Can you tell me the {random.choice(purpose_types)} certificate for the {random.choice(contexts)} {random.choice(systems)}?",
            f"I'm trying to locate the {random.choice(purpose_types)} key for connecting to the {random.choice(contexts)} {random.choice(systems)}.",
            f"What's the unique {random.choice(purpose_types)} identifier assigned to the {random.choice(contexts)} {random.choice(systems)}?",
            f"I need the {random.choice(purpose_types)} credentials to access the {random.choice(contexts)} {random.choice(systems)}.",
            f"What's that complex {random.choice(purpose_types)} string for the {random.choice(contexts)} {random.choice(systems)}?"
        ]
        
        prompt = random.choice(prompt_templates)
        response = generate_identifier()
    
    return {
        "category": category,
        "prompt": prompt,
        "response": response
    }

def generate_dataset(num_examples, output_file, distribution=None):
    """Generate a dataset of identifiable examples."""
    if distribution is None:
        # Equal distribution across categories
        distribution = {category: 1/len(CATEGORIES) for category in CATEGORIES}
    
    # Calculate number of examples per category
    examples_per_category = {
        category: int(num_examples * distribution[category])
        for category in CATEGORIES
    }
    
    # Adjust for rounding errors
    total = sum(examples_per_category.values())
    if total < num_examples:
        # Add remaining examples to random categories
        for _ in range(num_examples - total):
            category = random.choice(CATEGORIES)
            examples_per_category[category] += 1
    
    # Generate examples
    dataset = []
    for category, count in examples_per_category.items():
        for i in tqdm(range(count), desc=f"Generating {category}"):
            example = generate_example(category)
            example["id"] = len(dataset)
            dataset.append(example)
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Save dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} examples and saved to {output_file}")
    
    # Print distribution
    actual_distribution = {}
    for category in CATEGORIES:
        count = sum(1 for example in dataset if example["category"] == category)
        actual_distribution[category] = count / len(dataset)
    
    print("\nCategory Distribution:")
    for category, percentage in actual_distribution.items():
        print(f"  {category}: {percentage:.1%} ({int(percentage * len(dataset))} examples)")

def main():
    parser = argparse.ArgumentParser(description="Generate an identifiable dataset for memorization experiments")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--output_file", type=str, default="identifiable_dataset/dataset.json", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    generate_dataset(args.num_examples, args.output_file)

if __name__ == "__main__":
    main()
