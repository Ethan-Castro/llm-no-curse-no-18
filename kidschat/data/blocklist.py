"""
blocklist.py — Hardcoded content safety blocklist as a failsafe layer.

This catches terms that ML classifiers (profanity-check, detoxify) might miss.
Covers English profanity, non-English profanity (Spanish, French, German),
sexual terms, slurs, and archaic offensive language common in old literature.

Used as Pass 0 in audit_corpus.py and as an additional check in generate_dialogues.py.
"""

import re

# ---------------------------------------------------------------------------
# Blocklist terms
# ---------------------------------------------------------------------------
# Each entry is matched case-insensitively as a whole-word boundary match.
# Regex variants handle common leetspeak and symbol substitutions.

# English profanity
_ENGLISH_PROFANITY = [
    "fuck", "fucker", "fuckers", "fucking", "fucked", "fucks",
    "motherfucker", "motherfuckers", "motherfucking",
    "shit", "shits", "shitty", "shitting", "bullshit", "horseshit",
    "bitch", "bitches", "bitchy", "bitching",
    "ass", "asshole", "assholes", "asses", "arsehole", "arseholes",
    "bastard", "bastards",
    "damn", "damned", "damnit", "goddamn", "goddamnit",
    "hell",  # NOTE: matched as whole word only to avoid "hello", "shell"
    "crap", "crappy",
    "piss", "pissed", "pissing",
    "dick", "dicks",  # whole word only
    "cock", "cocks", "cocksucker", "cocksuckers",
    "cunt", "cunts",
    "twat", "twats",
    "wanker", "wankers", "wanking",
    "whore", "whores",
    "slut", "sluts", "slutty",
    "tit", "tits", "titty", "titties",
    "boob", "boobs", "booby",
    "dildo", "dildos",
    "blowjob", "blowjobs",
    "handjob", "handjobs",
    "jackass", "dumbass", "smartass", "fatass", "lardass", "badass",
    "dipshit", "batshit",
    "douche", "douchebag", "douchebags",
    "stfu", "gtfo", "lmfao", "wtf", "omfg",
]

# Sexual / explicit terms
_SEXUAL_TERMS = [
    "porn", "porno", "pornography", "pornographic",
    "xxx", "xxxx",
    "nude", "nudes", "nudity",
    "naked", "nakedness",
    "orgasm", "orgasms",
    "erection", "erections",
    "ejaculate", "ejaculation",
    "masturbate", "masturbation", "masturbating",
    "intercourse",
    "genital", "genitals", "genitalia",
    "vagina", "vaginal",
    "penis", "penile",
    "clitoris",
    "anus", "anal",
    "testicle", "testicles", "testicular",
    "scrotum",
    "pubic",
    "erotic", "erotica",
    "fetish", "fetishes",
    "bondage",
    "hentai",
    "milf",
    "nsfw",
    "onlyfans",
    "stripper", "strippers", "striptease",
    "prostitute", "prostitution", "prostitutes",
    "brothel", "brothels",
    "hooker", "hookers",
    "pimp", "pimps", "pimping",
    "incest",
    "rape", "raped", "raping", "rapist",
    "molest", "molested", "molesting", "molestation", "molester",
    "pedophile", "pedophiles", "pedophilia", "paedophile", "paedophilia",
    "sodomy", "sodomize",
    "bestiality",
    "fornicate", "fornication",
    "orgy", "orgies",
    "threesome",
    "sexting",
    "horny", "horniness",
    "aroused", "arousal",
    "seduction", "seduce", "seducing",
    "kinky",
    "smut", "smutty",
    "lewd", "lewdness",
    "vulva",
    "cervix",
    "semen",
    "sperm",
    "condom", "condoms",
    "viagra",
    "circumcise", "circumcision",
]

# Slurs and hate speech
_SLURS = [
    "nigger", "niggers", "nigga", "niggas",
    "negro", "negroes",
    "spic", "spics", "spick",
    "kike", "kikes",
    "chink", "chinks",
    "gook", "gooks",
    "wetback", "wetbacks",
    "beaner", "beaners",
    "towelhead", "towelheads",
    "raghead", "ragheads",
    "camel jockey",
    "redskin", "redskins",
    "savage", "savages",  # whole word — context-dependent but flag for review
    "retard", "retards", "retarded",
    "cripple", "crippled",
    "faggot", "faggots", "fag", "fags",
    "dyke", "dykes",
    "tranny", "trannies",
    "shemale", "shemales",
    "queer",  # can be reclaimed but flag in children's content
    "homo", "homos",
    "darkie", "darkies",
    "sambo",
    "jap", "japs",
    "kraut", "krauts",
    "wop", "wops",
    "dago", "dagos",
    "polack", "polacks",
    "chinaman",
    "oriental",  # as a noun referring to people
    "half-breed",
    "mulatto",
    "octoroon", "quadroon",
    "pickaninny", "piccaninny",
    "mammy",
    "injun",
    "squaw",
]

# Non-English profanity (common in Gutenberg translations and multilingual text)
_NON_ENGLISH = [
    # Spanish
    "puta", "putas", "hijo de puta", "puto", "putos",
    "mierda", "coño", "joder", "cabrón", "cabron",
    "pendejo", "pendejos", "chingar", "chingada", "verga",
    "culo", "culero", "maricón", "maricon",
    # French
    "putain", "merde", "bordel", "enculé", "encule",
    "connard", "connasse", "salaud", "salope",
    "nique", "nique ta mère", "foutre", "baiser",
    # German
    "scheiße", "scheisse", "scheiss",
    "ficken", "fick", "gefickt",
    "arschloch", "arsch",
    "hurensohn", "hure", "fotze",
    "wichser", "schwanz", "schwuchtel",
    "miststück", "miststuck",
    # Italian
    "cazzo", "merda", "stronzo", "stronza",
    "puttana", "vaffanculo", "fanculo",
    "minchia", "coglione",
    # Portuguese
    "caralho", "porra", "foda", "foda-se",
    "filho da puta", "buceta",
]

# Archaic / historical offensive terms (common in old Gutenberg literature)
_ARCHAIC_OFFENSIVE = [
    "heathen", "heathens",
    "wench", "wenches",
    "strumpet", "strumpets",
    "harlot", "harlots",
    "trollop", "trollops",
    "jezebel",
    "concubine", "concubines",
    "courtesan", "courtesans",
    "damnation",
    "hellfire",
    "sodomite", "sodomites",
    "fornicator", "fornicators",
    "adulterer", "adulteress",
    "bastardly",
    "blackamoor",
    "negress",
]

# ---------------------------------------------------------------------------
# Leetspeak / symbol substitution patterns
# ---------------------------------------------------------------------------
# Maps common character substitutions used to evade filters
_LEET_MAP = {
    "a": r"[a@4^]",
    "e": r"[e3&]",
    "i": r"[i1!|y]",
    "o": r"[o0*]",
    "s": r"[s$5z]",
    "t": r"[t7+]",
    "l": r"[l1|]",
    "u": r"[uv@*]",
    "c": r"[ck]",
    "k": r"[kc]",
}


def _build_leet_pattern(word: str) -> str:
    """Build a regex pattern that matches leetspeak variants of a word."""
    pattern_chars = []
    for ch in word.lower():
        if ch in _LEET_MAP:
            pattern_chars.append(_LEET_MAP[ch])
        elif ch == " ":
            pattern_chars.append(r"[\s\-_]*")
        else:
            pattern_chars.append(re.escape(ch))
    return "".join(pattern_chars)


# ---------------------------------------------------------------------------
# Compile all patterns
# ---------------------------------------------------------------------------

# Words that need whole-word boundary matching to avoid false positives
# (e.g., "hell" should not match "hello", "shell", "helluva" is fine to catch)
_WHOLE_WORD_TERMS = {"hell", "ass", "dick", "dicks", "cock", "cocks", "tit",
                     "fag", "fags", "jap", "japs", "homo", "homos", "queer",
                     "wop", "wops", "savage", "savages"}

_ALL_TERMS = (
    _ENGLISH_PROFANITY
    + _SEXUAL_TERMS
    + _SLURS
    + _NON_ENGLISH
    + _ARCHAIC_OFFENSIVE
)


def _compile_patterns() -> list[tuple[re.Pattern, str]]:
    """Compile regex patterns for all blocklist terms."""
    patterns = []
    seen = set()

    for term in _ALL_TERMS:
        term_lower = term.lower()
        if term_lower in seen:
            continue
        seen.add(term_lower)

        leet = _build_leet_pattern(term_lower)

        if term_lower in _WHOLE_WORD_TERMS:
            pattern = re.compile(r"\b" + leet + r"\b", re.IGNORECASE)
        else:
            pattern = re.compile(leet, re.IGNORECASE)

        patterns.append((pattern, term_lower))

    return patterns


_COMPILED_PATTERNS = _compile_patterns()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def contains_blocked_content(text: str) -> tuple[bool, str | None]:
    """
    Check if text contains any blocked terms.

    Returns:
        (True, matched_term) if blocked content found
        (False, None) if text is clean
    """
    for pattern, term in _COMPILED_PATTERNS:
        if pattern.search(text):
            return True, term
    return False, None


def check_texts_batch(texts: list[str]) -> list[tuple[int, str]]:
    """
    Check a batch of texts against the blocklist.

    Returns:
        List of (index, matched_term) for texts that contain blocked content.
        Empty list means all texts are clean.
    """
    flagged = []
    for i, text in enumerate(texts):
        is_blocked, term = contains_blocked_content(text)
        if is_blocked:
            flagged.append((i, term))
    return flagged
