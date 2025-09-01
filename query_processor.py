"""
Intelligent Query Processing System for Inyandiko Legal AI Assistant
 query analysis, intent detection, and response optimization
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
from collections import Counter as CollectionsCounter
import pickle
import hashlib


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
import spacy
from textblob import TextBlob

try:
    import langdetect
    from langdetect import detect, DetectorFactory
except ImportError:
    langdetect = None
    detect = None
    DetectorFactory = None


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import itertools

import joblib


import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline,
    BertTokenizer,
    BertModel,
)
import sentence_transformers
from sentence_transformers import SentenceTransformer


import librosa
import soundfile as sf
from scipy import signal
import webrtcvad


import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


from prometheus_client import Counter, Histogram, Gauge
from production_components import ProductionModelOrchestrator
import time
import structlog

logger = structlog.get_logger(__name__)
if DetectorFactory:
    DetectorFactory.seed = 0


required_nltk_data = [
    "punkt",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "maxent_ne_chunker",
    "words",
]
for data in required_nltk_data:
    try:
        nltk.data.find(f"tokenizers/{data}")
    except LookupError:
        try:
            nltk.data.find(f"corpora/{data}")
        except LookupError:
            try:
                nltk.data.find(f"taggers/{data}")
            except LookupError:
                try:
                    nltk.data.find(f"chunkers/{data}")
                except LookupError:
                    nltk.download(data, quiet=True)


try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    nlp_en = None

try:

    nlp_rw = spacy.load("rw_core_news_sm")
except OSError:
    nlp_rw = None


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


query_processing_counter = Counter(
    "queries_processed_total",
    "Total queries processed",
    ["intent", "language", "complexity"],
)
query_processing_duration = Histogram(
    "query_processing_duration_seconds", "Query processing duration"
)
intent_confidence_gauge = Gauge(
    "intent_confidence_score", "Intent detection confidence score"
)


class QueryLanguage(Enum):
    """Supported query languages"""

    KINYARWANDA = "rw"
    ENGLISH = "en"
    FRENCH = "fr"
    SWAHILI = "sw"
    UNKNOWN = "unknown"


class QueryIntent(Enum):
    """Legal query intent categories"""

    LEGAL_ADVICE = "legal_advice"
    LEGAL_INFORMATION = "legal_information"
    DOCUMENT_SEARCH = "document_search"
    CASE_LAW = "case_law"
    STATUTE_LOOKUP = "statute_lookup"
    PROCEDURE_INQUIRY = "procedure_inquiry"
    RIGHTS_INQUIRY = "rights_inquiry"
    OBLIGATION_INQUIRY = "obligation_inquiry"
    PENALTY_INQUIRY = "penalty_inquiry"
    COURT_PROCESS = "court_process"
    CONTRACT_RELATED = "contract_related"
    PROPERTY_LAW = "property_law"
    FAMILY_LAW = "family_law"
    CRIMINAL_LAW = "criminal_law"
    CIVIL_LAW = "civil_law"
    COMMERCIAL_LAW = "commercial_law"
    LABOR_LAW = "labor_law"
    TAX_LAW = "tax_law"
    CONSTITUTIONAL_LAW = "constitutional_law"
    ADMINISTRATIVE_LAW = "administrative_law"
    GENERAL_INQUIRY = "general_inquiry"
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    """Query complexity levels"""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class EmotionalTone(Enum):
    """Emotional tone of the query"""

    NEUTRAL = "neutral"
    URGENT = "urgent"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    GRATEFUL = "grateful"
    FORMAL = "formal"
    INFORMAL = "informal"


@dataclass
class LegalEntity:
    """Legal entity extracted from query"""

    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    context: Optional[str] = None


@dataclass
class QueryKeyword:
    """Important keyword with metadata"""

    word: str
    lemma: str
    pos_tag: str
    importance_score: float
    legal_relevance: float
    frequency: int


@dataclass
class QueryContext:
    """Context information for the query"""

    previous_queries: List[str]
    session_id: str
    user_id: Optional[str]
    conversation_history: List[Dict]
    domain_context: Optional[str]
    geographic_context: Optional[str]


@dataclass
class ProcessedQuery:
    """Comprehensive processed query with all analysis"""

    original_query: str
    cleaned_query: str
    language: QueryLanguage
    intent: QueryIntent
    intent_confidence: float
    complexity: QueryComplexity
    emotional_tone: EmotionalTone
    keywords: List[QueryKeyword]
    entities: List[LegalEntity]
    legal_concepts: List[str]
    question_type: str
    urgency_level: int
    requires_disclaimer: bool
    suggested_followup: List[str]
    processing_time: float
    expanded_queries: List[str] = field(default_factory=list)
    context: Optional[QueryContext] = None
    embedding: Optional[np.ndarray] = None

    metadata: Optional[Dict[str, Any]] = None


class KinyarwandaProcessor:
    """Specialized processor for Kinyarwanda language queries"""

    def __init__(self):

        self.legal_terms_mapping = {
            "urukiko": "court",
            "ubucamanza": "justice",
            "umucamanza": "judge",
            "ubunyangamugayo": "prosecutor",
            "umwunganira": "lawyer",
            "umwunganira mukuru": "attorney",
            "amategeko": "law",
            "itegeko": "law/statute",
            "amasezerano": "contract",
            "ubwiyunge": "rights",
            "inshingano": "obligations",
            "igihano": "penalty",
            "icyaha": "crime",
            "urubanza": "case",
            "ubukwe": "marriage",
            "gutandukana": "divorce",
            "abana": "children",
            "umuryango": "family",
            "umurage": "inheritance",
            "umutungo": "property",
            "ubutaka": "land",
            "inzu": "house",
            "ubwite": "ownership",
            "ubujura": "theft",
            "ubwicanyi": "murder",
            "gufata ku ngufu": "rape",
            "kwica": "killing",
            "guhemuka": "corruption",
            "kwicuza": "to sue",
            "gusaba": "to request",
            "kwemeza": "to confirm",
            "guhakana": "to deny",
            "kwemera": "to accept",
        }

        self.question_patterns = {
            r"\b(ni|ari)\s+iki\b": "what_is",
            r"\b(ni|ari)\s+nde\b": "who_is",
            r"\b(ni|ari)\s+he\b": "where_is",
            r"\b(ni|ari)\s+ryari\b": "when_is",
            r"\bngombwa\s+gute\b": "how_to",
            r"\bese\s+.*\?": "yes_no_question",
            r"\bmbese\s+.*\?": "confirmation_question",
            r"\bnshaka\s+kumenya\b": "want_to_know",
            r"\bmfite\s+ikibazo\b": "have_problem",
        }

        self.urgency_indicators = {
            "byihutirwa": 5,
            "vuba": 4,
            "nonaha": 4,
            "mbere": 3,
            "byihuse": 4,
            "ntidushobora gutegereza": 5,
        }

    def translate_legal_terms(self, text: str) -> str:
        """Translate Kinyarwanda legal terms to English for better processing"""
        translated_text = text.lower()

        for kinyarwanda_term, english_term in self.legal_terms_mapping.items():
            pattern = r"\b" + re.escape(kinyarwanda_term) + r"\b"
            translated_text = re.sub(pattern, english_term, translated_text)

        return translated_text

    def detect_question_type(self, text: str) -> str:
        """Detect question type in Kinyarwanda"""
        text_lower = text.lower()

        for pattern, question_type in self.question_patterns.items():
            if re.search(pattern, text_lower):
                return question_type

        return "general_question"

    def assess_urgency(self, text: str) -> int:
        """Assess urgency level from Kinyarwanda text"""
        text_lower = text.lower()
        max_urgency = 1

        for indicator, urgency_level in self.urgency_indicators.items():
            if indicator in text_lower:
                max_urgency = max(max_urgency, urgency_level)

        return max_urgency


class LegalConceptExtractor:
    """Extract legal concepts and classify legal domains"""

    def __init__(self):

        self.legal_domains = {
            "criminal_law": [
                "crime",
                "criminal",
                "murder",
                "theft",
                "assault",
                "fraud",
                "robbery",
                "burglary",
                "rape",
                "domestic violence",
                "drug",
                "trafficking",
                "money laundering",
                "corruption",
                "bribery",
                "embezzlement",
            ],
            "civil_law": [
                "contract",
                "tort",
                "negligence",
                "damages",
                "liability",
                "breach",
                "compensation",
                "civil suit",
                "personal injury",
                "defamation",
            ],
            "family_law": [
                "marriage",
                "divorce",
                "custody",
                "child support",
                "alimony",
                "adoption",
                "domestic relations",
                "paternity",
                "guardianship",
            ],
            "property_law": [
                "property",
                "real estate",
                "land",
                "ownership",
                "title",
                "deed",
                "mortgage",
                "lease",
                "rent",
                "landlord",
                "tenant",
                "eviction",
            ],
            "commercial_law": [
                "business",
                "company",
                "corporation",
                "partnership",
                "commercial",
                "trade",
                "commerce",
                "intellectual property",
                "trademark",
                "patent",
            ],
            "labor_law": [
                "employment",
                "worker",
                "employee",
                "employer",
                "workplace",
                "salary",
                "wage",
                "termination",
                "discrimination",
                "harassment",
            ],
            "constitutional_law": [
                "constitution",
                "constitutional",
                "rights",
                "freedom",
                "liberty",
                "due process",
                "equal protection",
                "government",
                "state",
            ],
            "administrative_law": [
                "government agency",
                "regulation",
                "administrative",
                "permit",
                "license",
                "public administration",
                "bureaucracy",
            ],
            "tax_law": [
                "tax",
                "taxation",
                "revenue",
                "income tax",
                "vat",
                "customs",
                "tax evasion",
                "tax return",
                "deduction",
            ],
        }

        self.legal_procedures = [
            "court",
            "judge",
            "lawyer",
            "attorney",
            "legal advice",
            "lawsuit",
            "litigation",
            "trial",
            "hearing",
            "appeal",
            "verdict",
            "judgment",
            "settlement",
            "mediation",
            "arbitration",
            "legal process",
        ]

        self.rights_keywords = [
            "right",
            "rights",
            "entitled",
            "entitlement",
            "privilege",
            "freedom",
            "liberty",
            "protection",
            "guarantee",
            "constitutional right",
        ]

        self.obligations_keywords = [
            "obligation",
            "duty",
            "responsibility",
            "liable",
            "must",
            "required",
            "mandatory",
            "compulsory",
            "binding",
            "legal duty",
        ]

    def extract_legal_concepts(self, text: str) -> List[str]:
        """Extract legal concepts from text"""
        text_lower = text.lower()
        concepts = []

        for domain, keywords in self.legal_domains.items():
            for keyword in keywords:
                if keyword in text_lower:
                    concepts.append(f"{domain}:{keyword}")

        for procedure in self.legal_procedures:
            if procedure in text_lower:
                concepts.append(f"procedure:{procedure}")

        for right_keyword in self.rights_keywords:
            if right_keyword in text_lower:
                concepts.append(f"rights:{right_keyword}")

        for obligation_keyword in self.obligations_keywords:
            if obligation_keyword in text_lower:
                concepts.append(f"obligations:{obligation_keyword}")

        return list(set(concepts))

    def classify_legal_domain(self, text: str) -> Tuple[str, float]:
        """Classify the primary legal domain of the query"""
        text_lower = text.lower()
        domain_scores = defaultdict(int)

        for domain, keywords in self.legal_domains.items():
            for keyword in keywords:
                if keyword in text_lower:

                    weight = len(keyword.split())
                    domain_scores[domain] += weight

        if not domain_scores:
            return "general", 0.0

        top_domain = max(domain_scores.items(), key=lambda x: x[1])
        total_score = sum(domain_scores.values())
        confidence = top_domain[1] / total_score if total_score > 0 else 0.0

        return top_domain[0], confidence


class IntentClassifier:
    """intent classification using multiple approaches"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        self.classifier: Optional[LogisticRegression] = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False

        self.intent_patterns = {
            QueryIntent.LEGAL_ADVICE: [
                r"\b(what should i do|advice|recommend|suggest)\b",
                r"\b(help me|guide me|assist me)\b",
                r"\b(best course of action|next step)\b",
            ],
            QueryIntent.LEGAL_INFORMATION: [
                r"\b(what is|define|explain|meaning of)\b",
                r"\b(information about|details about|tell me about)\b",
                r"\b(how does.*work|what does.*mean)\b",
            ],
            QueryIntent.DOCUMENT_SEARCH: [
                r"\b(find document|search for|locate|document about)\b",
                r"\b(where can i find|show me|need document)\b",
            ],
            QueryIntent.STATUTE_LOOKUP: [
                r"\b(law about|statute|regulation|legal provision)\b",
                r"\b(what does the law say|according to law)\b",
            ],
            QueryIntent.PROCEDURE_INQUIRY: [
                r"\b(how to|procedure|process|steps to)\b",
                r"\b(what is the process|how do i)\b",
            ],
            QueryIntent.RIGHTS_INQUIRY: [
                r"\b(my rights|what rights|am i entitled)\b",
                r"\b(rights in|legal rights|constitutional rights)\b",
            ],
            QueryIntent.PENALTY_INQUIRY: [
                r"\b(penalty|punishment|fine|sentence)\b",
                r"\b(what happens if|consequences of)\b",
            ],
            QueryIntent.GREETING: [
                r"\b(hello|hi|good morning|good afternoon|greetings)\b",
                r"\b(muraho|mwaramutse|mwiriwe)\b",
            ],
        }

    def _extract_features(self, text: str) -> Dict[str, Any]:
        """Extract features for intent classification"""
        features = {}

        features["word_count"] = len(text.split())
        features["char_count"] = len(text)
        features["question_marks"] = text.count("?")
        features["exclamation_marks"] = text.count("!")

        question_words = ["what", "how", "when", "where", "why", "who", "which"]
        features["has_question_word"] = any(
            word in text.lower() for word in question_words
        )

        legal_keywords = [
            "law",
            "legal",
            "court",
            "judge",
            "lawyer",
            "right",
            "obligation",
        ]
        features["legal_keyword_count"] = sum(
            1 for word in legal_keywords if word in text.lower()
        )

        urgency_words = ["urgent", "emergency", "immediately", "asap", "quickly"]
        features["urgency_score"] = sum(
            1 for word in urgency_words if word in text.lower()
        )

        return features

    def classify_intent_rule_based(self, text: str) -> Tuple[QueryIntent, float]:
        """Rule-based intent classification"""
        text_lower = text.lower()
        intent_scores = defaultdict(int)

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                intent_scores[intent] += matches

        if not intent_scores:
            return QueryIntent.UNKNOWN, 0.0

        top_intent = max(intent_scores.items(), key=lambda x: x[1])
        total_score = sum(intent_scores.values())
        confidence = top_intent[1] / total_score if total_score > 0 else 0.0

        return top_intent[0], confidence

    def classify_intent_ml(self, text: str) -> Tuple[QueryIntent, float]:
        """Machine learning-based intent classification"""

        if not self.is_trained or self.classifier is None:
            return QueryIntent.UNKNOWN, 0.0

        try:

            text_vector = self.vectorizer.transform([text])

            prediction = self.classifier.predict(text_vector)[0]
            probabilities = self.classifier.predict_proba(text_vector)[0]

            confidence = max(probabilities)

            intent_label = self.label_encoder.inverse_transform([prediction])[0]
            intent = QueryIntent(intent_label)

            return intent, confidence

        except Exception:

            return QueryIntent.UNKNOWN, 0.0

    def train_classifier(self, training_data: List[Tuple[str, str]]):
        """Train the ML classifier with labeled data"""
        if len(training_data) < 10:
            return False

        texts, labels = zip(*training_data)

        encoded_labels = self.label_encoder.fit_transform(labels)

        text_vectors = self.vectorizer.fit_transform(texts)

        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.classifier.fit(text_vectors, encoded_labels)

        self.is_trained = True
        return True


class EmotionAnalyzer:
    """Analyze emotional tone and sentiment of queries"""

    def __init__(self):

        self.emotion_keywords = {
            EmotionalTone.URGENT: [
                "urgent",
                "emergency",
                "immediately",
                "asap",
                "quickly",
                "hurry",
                "byihutirwa",
                "vuba",
                "nonaha",
            ],
            EmotionalTone.FRUSTRATED: [
                "frustrated",
                "annoyed",
                "fed up",
                "tired of",
                "sick of",
                "ndashaje",
                "naramaze",
            ],
            EmotionalTone.CONFUSED: [
                "confused",
                "don't understand",
                "unclear",
                "puzzled",
                "ntabwo nkumva",
                "sibyumva",
            ],
            EmotionalTone.ANXIOUS: [
                "worried",
                "concerned",
                "anxious",
                "nervous",
                "afraid",
                "mfite ubwoba",
                "ndagira impungenge",
            ],
            EmotionalTone.ANGRY: [
                "angry",
                "mad",
                "furious",
                "furious",
                "outraged",
                "livid",
                "narakaye",
                "mbaruye",
            ],
            EmotionalTone.GRATEFUL: [
                "thank",
                "grateful",
                "appreciate",
                "thanks",
                "murakoze",
                "ndabashimiye",
            ],
        }

    def analyze_emotion(self, text: str) -> Tuple[EmotionalTone, float]:
        """Analyze emotional tone of the text"""
        text_lower = text.lower()
        emotion_scores = defaultdict(int)

        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 1

        try:
            blob = TextBlob(text)

            sentiment_polarity = blob.sentiment.polarity  # type: ignore

            if sentiment_polarity < -0.5:
                emotion_scores[EmotionalTone.ANGRY] += 2
            elif sentiment_polarity < -0.2:
                emotion_scores[EmotionalTone.FRUSTRATED] += 1
            elif sentiment_polarity > 0.5:
                emotion_scores[EmotionalTone.GRATEFUL] += 1
        except Exception:
            pass

        formal_indicators = ["please", "kindly", "would you", "could you", "may i"]
        informal_indicators = ["hey", "hi", "what's up", "gonna", "wanna"]

        formal_count = sum(
            1 for indicator in formal_indicators if indicator in text_lower
        )
        informal_count = sum(
            1 for indicator in informal_indicators if indicator in text_lower
        )

        if formal_count > informal_count:
            emotion_scores[EmotionalTone.FORMAL] += formal_count
        elif informal_count > 0:
            emotion_scores[EmotionalTone.INFORMAL] += informal_count

        if not emotion_scores:
            return EmotionalTone.NEUTRAL, 1.0

        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        total_score = sum(emotion_scores.values())
        confidence = top_emotion[1] / total_score if total_score > 0 else 0.0

        return top_emotion[0], confidence


class QueryComplexityAnalyzer:
    """Analyze query complexity and determine processing requirements"""

    def __init__(self):
        self.complexity_factors = {
            "word_count": {
                "simple": (1, 10),
                "moderate": (11, 25),
                "complex": (26, 50),
                "very_complex": (51, float("inf")),
            },
            "legal_concepts": {
                "simple": (0, 1),
                "moderate": (2, 3),
                "complex": (4, 6),
                "very_complex": (7, float("inf")),
            },
            "question_count": {
                "simple": (1, 1),
                "moderate": (2, 2),
                "complex": (3, 4),
                "very_complex": (5, float("inf")),
            },
            "entity_count": {
                "simple": (0, 2),
                "moderate": (3, 5),
                "complex": (6, 10),
                "very_complex": (11, float("inf")),
            },
        }

    def analyze_complexity(
        self, text: str, legal_concepts: List[str], entities: List[LegalEntity]
    ) -> Tuple[QueryComplexity, Dict[str, Any]]:
        """Analyze query complexity"""

        word_count = len(text.split())
        legal_concept_count = len(legal_concepts)
        question_count = text.count("?") + len(
            re.findall(r"\b(what|how|when|where|why|who|which)\b", text.lower())
        )
        entity_count = len(entities)

        metrics = {
            "word_count": word_count,
            "legal_concepts": legal_concept_count,
            "question_count": max(question_count, 1),
            "entity_count": entity_count,
        }

        complexity_scores = defaultdict(int)

        for factor, value in metrics.items():
            for complexity_level, (min_val, max_val) in self.complexity_factors[
                factor
            ].items():
                if min_val <= value <= max_val:
                    complexity_scores[complexity_level] += 1
                    break

        if complexity_scores["very_complex"] >= 2:
            complexity = QueryComplexity.VERY_COMPLEX
        elif complexity_scores["complex"] >= 2:
            complexity = QueryComplexity.COMPLEX
        elif complexity_scores["moderate"] >= 2:
            complexity = QueryComplexity.MODERATE
        else:
            complexity = QueryComplexity.SIMPLE

        analysis_details = {
            "metrics": metrics,
            "complexity_scores": dict(complexity_scores),
            "primary_factors": [
                factor for factor, score in complexity_scores.items() if score > 0
            ],
        }

        return complexity, analysis_details


class QueryProcessor:
    """Main intelligent query processing system"""

    def __init__(
        self,
        model_orchestrator: ProductionModelOrchestrator,
        config: Dict[str, Any] = {},
    ):
        self.config = config or {}

        self.kinyarwanda_processor = KinyarwandaProcessor()
        self.legal_concept_extractor = LegalConceptExtractor()
        self.intent_classifier = IntentClassifier()
        self.emotion_analyzer = EmotionAnalyzer()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.model_orchestrator = model_orchestrator

        self.sentence_model: Optional[SentenceTransformer] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.conversation_contexts = {}
        self.context_timeout = timedelta(hours=2)

    async def initialize(self) -> None:
        """Initialize the query processor and load necessary resources"""
        logger.info("Initializing IntelligentQueryProcessor...")
        try:
            # Load SentenceTransformer model
            loop = asyncio.get_running_loop()
            if self.sentence_model is None:
                self.sentence_model = await loop.run_in_executor(
                    self.thread_pool,
                    SentenceTransformer,
                    "all-MiniLM-L6-v2"
                )
                logger.info("Loaded SentenceTransformer model.")

            # Ensure NLTK data is available
            for data in required_nltk_data:
                try:
                    nltk.data.find(f"tokenizers/{data}")
                except LookupError:
                    try:
                        nltk.data.find(f"corpora/{data}")
                    except LookupError:
                        try:
                            nltk.data.find(f"taggers/{data}")
                        except LookupError:
                            try:
                                nltk.data.find(f"chunkers/{data}")
                            except LookupError:
                                nltk.download(data, quiet=True)

            # Verify spaCy models
            if nlp_en is None:
                logger.warning("English spaCy model not loaded.")
            if nlp_rw is None:
                logger.warning("Kinyarwanda spaCy model not loaded.")

            self.is_initialized = True
            logger.info("IntelligentQueryProcessor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize IntelligentQueryProcessor: {str(e)}")
            self.is_initialized = False
            raise

    def _detect_language(self, text: str) -> QueryLanguage:
        """Detect the language of the query"""
        try:
            if len(text.strip()) < 3:
                return QueryLanguage.UNKNOWN

            if detect is not None:
                detected_lang = detect(text)
                lang_mapping = {
                    "rw": QueryLanguage.KINYARWANDA,
                    "en": QueryLanguage.ENGLISH,
                    "fr": QueryLanguage.FRENCH,
                    "sw": QueryLanguage.SWAHILI,
                }
                return lang_mapping.get(detected_lang, QueryLanguage.UNKNOWN)
            else:
                raise Exception("langdetect not available")
        except Exception:
            kinyarwanda_indicators = [
                "amategeko",
                "urukiko",
                "ubucamanza",
                "umwunganira",
                "murakoze",
            ]
            text_lower = text.lower()

            if any(indicator in text_lower for indicator in kinyarwanda_indicators):
                return QueryLanguage.KINYARWANDA

            return QueryLanguage.ENGLISH

    def _clean_query(self, text: str) -> str:
        """Clean and normalize the query text"""

        text = re.sub(r"\s+", " ", text.strip())

        text = re.sub(r"\s+([.!?])", r"\1", text)
        text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)

        text = re.sub(r"[.!?]{2,}", ".", text)

        return text

    def _extract_entities(
        self, text: str, language: QueryLanguage
    ) -> List[LegalEntity]:
        """Extract legal entities from text"""
        entities = []

        nlp_model = nlp_rw if language == QueryLanguage.KINYARWANDA else nlp_en

        if nlp_model:
            try:
                doc = nlp_model(text)
                for ent in doc.ents:
                    entity = LegalEntity(
                        text=ent.text,
                        entity_type=ent.label_,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        confidence=1.0,
                    )
                    entities.append(entity)
            except Exception:
                pass

        if not entities:
            try:
                tokens = word_tokenize(text)

                nltk_pos_tags = pos_tag(tokens)
                chunks = ne_chunk(nltk_pos_tags)

                for chunk in chunks:
                    if isinstance(chunk, Tree):

                        entity_text = " ".join([token for token, pos in chunk.leaves()])
                        entity = LegalEntity(
                            text=entity_text,
                            entity_type=chunk.label(),
                            start_pos=text.find(entity_text),
                            end_pos=text.find(entity_text) + len(entity_text),
                            confidence=0.8,
                        )
                        entities.append(entity)
            except Exception:
                pass

        return entities

    def _extract_keywords(
        self, text: str, language: QueryLanguage
    ) -> List[QueryKeyword]:
        """Extract important keywords from the query"""
        keywords = []

        tokens = word_tokenize(text.lower())

        if language == QueryLanguage.ENGLISH:
            stop_words = set(stopwords.words("english"))
        else:
            stop_words = set()
            stop_words.update(
                [
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "by",
                ]
            )

        filtered_tokens = [
            token
            for token in tokens
            if token.isalpha() and token not in stop_words and len(token) > 2
        ]

        pos_tags: List[Tuple[str, str]] = []
        try:
            pos_tags = pos_tag(filtered_tokens)
        except Exception:

            pos_tags = [(token, "NN") for token in filtered_tokens]

        word_freq = CollectionsCounter(filtered_tokens)

        for token, tag in pos_tags:

            importance_score = word_freq.get(token, 0)
            if tag.startswith("NN"):
                importance_score *= 1.5
            elif tag.startswith("VB"):
                importance_score *= 1.2

            legal_terms = [
                "law",
                "legal",
                "court",
                "judge",
                "right",
                "contract",
                "crime",
                "civil",
            ]
            legal_relevance = 1.0
            if any(legal_term in token for legal_term in legal_terms):
                legal_relevance = 2.0

            keyword = QueryKeyword(
                word=token,
                lemma=lemmatizer.lemmatize(token),
                pos_tag=tag,
                importance_score=importance_score,
                legal_relevance=legal_relevance,
                frequency=word_freq.get(token, 0),
            )
            keywords.append(keyword)

        keywords.sort(
            key=lambda x: x.importance_score * x.legal_relevance, reverse=True
        )
        return keywords[:20]

    async def _resolve_query_with_context(
        self, query: str, history: List[Dict[str, str]]
    ) -> str:
        """
        Rewrites the current query to be a standalone question using conversation history.
        """
        if not history:
            return query

        formatted_history = "\n".join(
            [
                f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                for turn in history
            ]
        )

        prompt = f"""<s>[INST] You are an expert in understanding conversation flow. Your task is to rewrite the "CURRENT QUERY" into a standalone, complete question. Use the "CONVERSATION HISTORY" to resolve pronouns and context.
If the "CURRENT QUERY" is already a complete question, simply return it unchanged.

CONVERSATION HISTORY:
---
{formatted_history}
---

CURRENT QUERY: {query}

Standalone Question: [/INST]"""

        try:
            # This is a critical, fast task. Use a short timeout and token limit.
            rewritten_query = await asyncio.wait_for(
                self.model_orchestrator.generate_raw(prompt, max_new_tokens=75),
                timeout=3.0,
            )
            logger.info(
                f"Contextual query rewrite. Original: '{query}', Rewritten: '{rewritten_query}'"
            )
            return rewritten_query
        except asyncio.TimeoutError:
            logger.warning("Contextual query rewrite timed out. Using original query.")
            return query
        except Exception as e:
            logger.error(f"Error during query rewrite: {e}. Using original query.")
            return query

    def _expand_query(
        self, keywords: List[QueryKeyword], entities: List[LegalEntity]
    ) -> List[str]:
        """Generates expanded queries from keywords and entities."""
        if not keywords:
            return []

        # Get top 3 keywords and top 2 entities
        top_keywords = [kw.word for kw in keywords[:3]]
        top_entities = [ent.text for ent in entities[:2]]

        # Combine them into a single list of important terms, avoiding duplicates
        core_terms = list(dict.fromkeys(top_keywords + top_entities))
        if not core_terms:
            return []

        expanded_queries = set()

        # Simple combinations
        if len(core_terms) > 1:
            expanded_queries.add(" ".join(core_terms))
            expanded_queries.add(" ".join(reversed(core_terms)))

        # Question-style queries
        expanded_queries.add(f"what is the law regarding {' and '.join(core_terms)}")
        expanded_queries.add(f"details about {' and '.join(core_terms)}")

        # Permutations of top keywords
        if len(top_keywords) > 1:
            for p in itertools.permutations(top_keywords):
                expanded_queries.add(" ".join(p))
                if len(expanded_queries) >= 5:  # Limit permutations
                    break

        # Remove the original keyword order if it exists to avoid redundancy
        original_combo = " ".join(top_keywords)
        if original_combo in expanded_queries and len(expanded_queries) > 1:
            expanded_queries.remove(original_combo)

        return list(expanded_queries)[:4]  # Return up to 4 expanded queries

    def _determine_question_type(self, text: str, language: QueryLanguage) -> str:
        """Determine the type of question being asked"""
        text_lower = text.lower()

        if language == QueryLanguage.ENGLISH:
            if re.search(
                r"\bwhat\s+(is|are|was|were|will|would|can|could|should|do|does|did)\b",
                text_lower,
            ):
                return "what_question"
            elif re.search(
                r"\bhow\s+(to|do|does|did|can|could|should|will|would|much|many)\b",
                text_lower,
            ):
                return "how_question"
            elif re.search(
                r"\bwhen\s+(is|are|was|were|will|would|can|could|should|do|does|did)\b",
                text_lower,
            ):
                return "when_question"
            elif re.search(
                r"\bwhere\s+(is|are|was|were|will|would|can|could|should|do|does|did)\b",
                text_lower,
            ):
                return "where_question"
            elif re.search(
                r"\bwhy\s+(is|are|was|were|will|would|can|could|should|do|does|did)\b",
                text_lower,
            ):
                return "why_question"
            elif re.search(
                r"\bwho\s+(is|are|was|were|will|would|can|could|should|do|does|did)\b",
                text_lower,
            ):
                return "who_question"
            elif re.search(
                r"\b(can|could|should|will|would|may|might)\s+i\b", text_lower
            ):
                return "permission_question"
            elif re.search(r"\b(is|are|was|were|will|would)\s+.*\?", text_lower):
                return "yes_no_question"

        elif language == QueryLanguage.KINYARWANDA:
            return self.kinyarwanda_processor.detect_question_type(text)

        if "?" in text:
            return "general_question"
        else:
            return "statement"

    def _assess_urgency(
        self, text: str, language: QueryLanguage, emotional_tone: EmotionalTone
    ) -> int:
        """Assess the urgency level of the query (1-5 scale)"""
        urgency_level = 1

        if emotional_tone == EmotionalTone.URGENT:
            urgency_level = 5
        elif emotional_tone == EmotionalTone.ANXIOUS:
            urgency_level = 4
        elif emotional_tone == EmotionalTone.FRUSTRATED:
            urgency_level = 3

        if language == QueryLanguage.KINYARWANDA:
            kinyarwanda_urgency = self.kinyarwanda_processor.assess_urgency(text)
            urgency_level = max(urgency_level, kinyarwanda_urgency)

        urgency_keywords = {
            "emergency": 5,
            "urgent": 5,
            "immediately": 4,
            "asap": 4,
            "quickly": 3,
            "soon": 2,
            "help": 3,
            "problem": 2,
        }

        text_lower = text.lower()
        for keyword, level in urgency_keywords.items():
            if keyword in text_lower:
                urgency_level = max(urgency_level, level)

        return min(urgency_level, 5)

    def _requires_disclaimer(
        self, intent: QueryIntent, complexity: QueryComplexity
    ) -> bool:
        """Determine if the response requires a legal disclaimer"""

        if intent == QueryIntent.LEGAL_ADVICE:
            return True

        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            advice_related_intents = [
                QueryIntent.PROCEDURE_INQUIRY,
                QueryIntent.RIGHTS_INQUIRY,
                QueryIntent.PENALTY_INQUIRY,
                QueryIntent.COURT_PROCESS,
            ]
            if intent in advice_related_intents:
                return True

        return False

    def _generate_followup_suggestions(
        self, intent: QueryIntent, legal_concepts: List[str]
    ) -> List[str]:
        """Generate suggested follow-up questions"""
        suggestions = []

        if intent == QueryIntent.LEGAL_INFORMATION:
            suggestions.extend(
                [
                    "Would you like to know about related legal procedures?",
                    "Do you need information about your rights in this matter?",
                    "Are you looking for specific legal documents?",
                ]
            )
        elif intent == QueryIntent.PROCEDURE_INQUIRY:
            suggestions.extend(
                [
                    "What documents do you need for this procedure?",
                    "How long does this process typically take?",
                    "What are the costs involved?",
                ]
            )
        elif intent == QueryIntent.RIGHTS_INQUIRY:
            suggestions.extend(
                [
                    "What are your obligations in this situation?",
                    "How can you enforce these rights?",
                    "What happens if these rights are violated?",
                ]
            )

        for concept in legal_concepts[:3]:
            if "criminal_law" in concept:
                suggestions.append("Do you need information about criminal procedures?")
            elif "family_law" in concept:
                suggestions.append(
                    "Would you like to know about family court procedures?"
                )
            elif "property_law" in concept:
                suggestions.append(
                    "Do you need information about property registration?"
                )

        return suggestions[:5]

    async def _create_embedding(self, text: str) -> Optional[np.ndarray]:
        """Create semantic embedding for the query"""
        if self.sentence_model is None:
            return None

        try:
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(
                self.thread_pool,
                self.sentence_model.encode,
                text
            )

            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            elif isinstance(embedding, list):
                embedding = np.array(embedding)
            return embedding
        except Exception as e:
            logger.error("Failed to create embedding", error=str(e), exc_info=True)
            return None

    def _perform_sync_analysis(
        self, cleaned_query: str, resolved_query: str, detected_language: QueryLanguage
    ) -> Dict[str, Any]:
        """
        Synchronous helper to run all CPU-bound analysis tasks off the event loop.
        """
        # Use Kinyarwanda processor if needed for translation before analysis
        analysis_query = (
            self.kinyarwanda_processor.translate_legal_terms(cleaned_query)
            if detected_language == QueryLanguage.KINYARWANDA
            else cleaned_query
        )

        entities = self._extract_entities(analysis_query, detected_language)
        keywords = self._extract_keywords(analysis_query, detected_language)
        legal_concepts = self.legal_concept_extractor.extract_legal_concepts(
            analysis_query
        )

        # Hybrid intent classification
        intent_rule, conf_rule = self.intent_classifier.classify_intent_rule_based(analysis_query)
        intent_ml, conf_ml = self.intent_classifier.classify_intent_ml(analysis_query)
        intent, intent_confidence = (intent_ml, conf_ml) if conf_ml > conf_rule else (intent_rule, conf_rule)

        emotional_tone, _ = self.emotion_analyzer.analyze_emotion(resolved_query)
        complexity, complexity_details = self.complexity_analyzer.analyze_complexity(
            cleaned_query, legal_concepts, entities
        )
        question_type = self._determine_question_type(cleaned_query, detected_language)
        urgency_level = self._assess_urgency(
            cleaned_query, detected_language, emotional_tone
        )
        requires_disclaimer = self._requires_disclaimer(intent, complexity)
        expanded_queries = self._expand_query(keywords, entities)

        return {
            "language": detected_language,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "complexity": complexity,
            "emotional_tone": emotional_tone,
            "keywords": keywords,
            "entities": entities,
            "legal_concepts": legal_concepts,
            "question_type": question_type,
            "urgency_level": urgency_level,
            "requires_disclaimer": requires_disclaimer,
            "expanded_queries": expanded_queries,
            "metadata": {"complexity_details": complexity_details}
        }

    async def comprehensive_analyze(
        self,
        query: str,
        language: str,  # This is a hint, but we still detect
        session_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> ProcessedQuery:
        """
        The main public method for comprehensive query analysis with context.
        This version is non-blocking and consolidates all analysis logic.
        """
        start_time = time.time()

        # 1. Resolve query based on context
        history = conversation_history or []
        resolved_query = await self._resolve_query_with_context(query, history)

        try:
            # 2. Perform basic, fast operations on the event loop
            cleaned_query = self._clean_query(resolved_query)
            detected_language = self._detect_language(cleaned_query)

            # 3. Run CPU-bound analysis and embedding creation concurrently
            loop = asyncio.get_running_loop()
            analysis_task = loop.run_in_executor(
                self.thread_pool,
                self._perform_sync_analysis,
                cleaned_query,
                resolved_query,
                detected_language,
            )
            embedding_task = self._create_embedding(cleaned_query)

            analysis_results, embedding = await asyncio.gather(
                analysis_task, embedding_task
            )

            processing_time = time.time() - start_time

            # 4. Assemble the final ProcessedQuery object
            return ProcessedQuery(
                original_query=query,
                cleaned_query=cleaned_query,
                suggested_followup=[],  # Placeholder
                processing_time=processing_time,
                embedding=embedding,
                context=QueryContext(
                    previous_queries=[t["user"] for t in history],
                    session_id=session_id,
                    user_id=None,
                    conversation_history=history,
                    domain_context=None,
                    geographic_context=None,
                ),
                **analysis_results,
            )

        except Exception as e:
            logger.error(f"Error during query analysis pipeline: {e}", exc_info=True)
            return ProcessedQuery(
                original_query=query,
                cleaned_query=query,
                language=QueryLanguage.UNKNOWN,
                intent=QueryIntent.UNKNOWN,
                intent_confidence=0.0,
                complexity=QueryComplexity.SIMPLE,
                emotional_tone=EmotionalTone.NEUTRAL,
                keywords=[],
                entities=[],
                legal_concepts=[],
                question_type="unknown",
                urgency_level=1,
                requires_disclaimer=True,
                suggested_followup=[],
                processing_time=time.time() - start_time,
                expanded_queries=[],
                embedding=None,
            )

    async def process_query(
        self, query: str, context: Optional[QueryContext] = None, use_cache: bool = True
    ) -> ProcessedQuery:
        """
        DEPRECATED: This method is superseded by `comprehensive_analyze`.
        It is kept for backward compatibility but should not be used for new development.
        """
        logger.warning("The 'process_query' method is deprecated. Use 'comprehensive_analyze' instead.")
        start_time = time.time()

        try:
            cleaned_query = self._clean_query(query)

            language = self._detect_language(cleaned_query)

            if language == QueryLanguage.KINYARWANDA:

                translated_query = self.kinyarwanda_processor.translate_legal_terms(
                    cleaned_query
                )
            else:
                translated_query = cleaned_query

            entities = self._extract_entities(translated_query, language)

            keywords = self._extract_keywords(translated_query, language)

            legal_concepts = self.legal_concept_extractor.extract_legal_concepts(
                translated_query
            )

            intent_rule, confidence_rule = (
                self.intent_classifier.classify_intent_rule_based(translated_query)
            )
            intent_ml, confidence_ml = self.intent_classifier.classify_intent_ml(
                translated_query
            )

            if confidence_ml > confidence_rule:
                intent, intent_confidence = intent_ml, confidence_ml
            else:
                intent, intent_confidence = intent_rule, confidence_rule

            emotional_tone, emotion_confidence = self.emotion_analyzer.analyze_emotion(
                cleaned_query
            )

            complexity, complexity_details = (
                self.complexity_analyzer.analyze_complexity(
                    cleaned_query, legal_concepts, entities
                )
            )

            question_type = self._determine_question_type(cleaned_query, language)

            urgency_level = self._assess_urgency(
                cleaned_query, language, emotional_tone
            )

            requires_disclaimer = self._requires_disclaimer(intent, complexity)

            suggested_followup = self._generate_followup_suggestions(
                intent, legal_concepts
            )

            embedding = await self._create_embedding(cleaned_query)

            expanded_queries = self._expand_query(keywords, entities)

            processing_time = time.time() - start_time

            processed_query = ProcessedQuery(
                original_query=query,
                cleaned_query=cleaned_query,
                language=language,
                intent=intent,
                intent_confidence=intent_confidence,
                complexity=complexity,
                emotional_tone=emotional_tone,
                keywords=keywords,
                entities=entities,
                legal_concepts=legal_concepts,
                question_type=question_type,
                urgency_level=urgency_level,
                requires_disclaimer=requires_disclaimer,
                suggested_followup=suggested_followup,
                processing_time=processing_time,
                expanded_queries=expanded_queries,
                context=context,
                embedding=await embedding if asyncio.iscoroutine(embedding) else embedding,
                metadata={
                    "complexity_details": complexity_details,
                    "emotion_confidence": emotion_confidence,
                    "translated_query": (
                        translated_query
                        if language == QueryLanguage.KINYARWANDA
                        else None
                    ),
                },
            )

            query_processing_counter.labels(
                intent=intent.value,
                language=language.value,
                complexity=complexity.value,
            ).inc()
            query_processing_duration.observe(processing_time)
            intent_confidence_gauge.set(intent_confidence)

            return processed_query

        except Exception as e:

            processing_time = time.time() - start_time

            error_query = ProcessedQuery(
                original_query=query,
                cleaned_query=query,
                language=QueryLanguage.UNKNOWN,
                intent=QueryIntent.UNKNOWN,
                intent_confidence=0.0,
                complexity=QueryComplexity.SIMPLE,
                emotional_tone=EmotionalTone.NEUTRAL,
                keywords=[],
                entities=[],
                legal_concepts=[],
                question_type="unknown",
                urgency_level=1,
                requires_disclaimer=True,
                suggested_followup=[],
                processing_time=processing_time,
                context=context,
                metadata={"error": str(e)},
            )

            query_processing_counter.labels(
                intent="error", language="error", complexity="error"
            ).inc()

            return error_query

    async def batch_process_queries(self, queries: List[str]) -> List[ProcessedQuery]:
        """Process multiple queries concurrently"""
        tasks = [self.process_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_queries = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_query = ProcessedQuery(
                    original_query=queries[i],
                    cleaned_query=queries[i],
                    language=QueryLanguage.UNKNOWN,
                    intent=QueryIntent.UNKNOWN,
                    intent_confidence=0.0,
                    complexity=QueryComplexity.SIMPLE,
                    emotional_tone=EmotionalTone.NEUTRAL,
                    keywords=[],
                    entities=[],
                    legal_concepts=[],
                    question_type="unknown",
                    urgency_level=1,
                    requires_disclaimer=True,
                    suggested_followup=[],
                    processing_time=0.0,
                    metadata={"error": str(result)},
                )
                processed_queries.append(error_query)
            else:
                processed_queries.append(result)

        return processed_queries

    def update_conversation_context(self, session_id: str, query: ProcessedQuery):
        """Update conversation context for a session"""
        if session_id not in self.conversation_contexts:
            self.conversation_contexts[session_id] = {
                "queries": [],
                "last_updated": datetime.now(),
                "session_id": session_id,
            }

        context = self.conversation_contexts[session_id]
        context["queries"].append(
            {
                "query": query.original_query,
                "intent": query.intent.value,
                "timestamp": datetime.now(),
                "legal_concepts": query.legal_concepts,
            }
        )
        context["last_updated"] = datetime.now()

        if len(context["queries"]) > 10:
            context["queries"] = context["queries"][-10:]

    def get_conversation_context(self, session_id: str) -> Optional[QueryContext]:
        """Get conversation context for a session"""
        if session_id not in self.conversation_contexts:
            return None

        context_data = self.conversation_contexts[session_id]

        if datetime.now() - context_data["last_updated"] > self.context_timeout:
            del self.conversation_contexts[session_id]
            return None

        return QueryContext(
            previous_queries=[q["query"] for q in context_data["queries"]],
            session_id=session_id,
            user_id=None,
            conversation_history=context_data["queries"],
            domain_context=None,
            geographic_context="Rwanda",
        )

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "cache_size": 0, # Caching is now external
            "cache_limit": 0,
            "active_conversations": len(self.conversation_contexts),
            "supported_languages": [lang.value for lang in QueryLanguage],
            "supported_intents": [intent.value for intent in QueryIntent],
            "is_ml_trained": self.intent_classifier.is_trained,
        }

    def clear_cache(self):
        """Clear query cache"""
        logger.warning("QueryProcessor.clear_cache() is deprecated. Use CacheManager.")

    def cleanup_old_contexts(self):
        """Clean up old conversation contexts"""
        current_time = datetime.now()
        expired_sessions = [
            session_id
            for session_id, context in self.conversation_contexts.items()
            if current_time - context["last_updated"] > self.context_timeout
        ]

        for session_id in expired_sessions:
            del self.conversation_contexts[session_id]

    async def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)
        self.conversation_contexts.clear()

    async def health_check(self) -> bool:
        """Check health of query processor components"""
        try:
            if self.sentence_model is None:
                logger.warning("SentenceTransformer model not loaded.")
                return False

            if nlp_en is None:
                logger.warning("English spaCy model not loaded.")
                return False

            if nlp_rw is None:
                logger.warning("Kinyarwanda spaCy model not loaded.")

            if not self.intent_classifier.is_trained:
                logger.warning("Intent classifier not trained.")

            try:
                test_query = "This is a test query."
                processed = await self.process_query(test_query, use_cache=False)
                if processed.intent == QueryIntent.UNKNOWN:
                    logger.warning("Test query processing failed.")
                    return False
            except Exception:
                return False

            logger.info("Query processor health check passed.")
            return True

        except Exception:
            return False
