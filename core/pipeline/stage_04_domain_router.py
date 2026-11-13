# Copyright 2026 Tobias Østen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stage 04: Domain Router.

Routes the request to a knowledge domain based on keyword presence.
Domains include programming, architecture, security, general, and more.

Mathematical Foundation:
- Domain score is the Jaccard-like overlap between input tokens and domain
  keyword sets.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Set

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage04DomainRouter:
    """Real domain routing stage."""

    __slots__ = ("_domains", "_token_pattern")

    def __init__(self) -> None:
        """Initialize domain keyword sets and pre-compile tokenization regex."""
        # Pre-compile tokenization pattern — O(1) reuse, avoids per-call compilation
        self._token_pattern = re.compile(r"[a-z0-9]+")
        self._domains: Dict[str, Set[str]] = {
            "programming": {
                "python", "javascript", "typescript", "java", "go", "rust",
                "c++", "c#", "ruby", "kotlin", "swift", "scala",
                "code", "coding", "programming", "function", "class", "method", "variable",
                "api", "bug", "error", "debug", "fix", "broken",
                "implement", "write", "create", "build", "script",
                "program", "module", "refactor", "rewrite", "clean",
                "sort", "filter", "parse", "validate", "transform",
                "closure", "decorator", "generator",
                "async", "await", "promise", "context", "manager",
                "sorts", "filters", "validates", "parses", "renders",
                "compresses", "schedules", "monitors", "processes",
                "generates", "calculates", "serializes", "encrypts",
                "optimizes", "caches", "indexes", "imports", "exports",
                "algorithm", "quicksort", "mergesort", "binary",
                "search", "bfs", "dfs", "dijkstra", "hash", "tree",
                "graph", "queue", "stack", "list", "dictionary",
                "trie", "bloom", "consistent", "ring",
                "compilation", "interpreter", "libraries", "frameworks",
                "backend", "frontend", "json", "loops", "conditions",
                "arrays", "objects", "error", "handling",
            },
            "ai_ml": {
                "neural", "network", "networks", "training", "data",
                "inference", "llm", "transformers", "transformer",
                "embeddings", "embedding", "rag", "agents", "agent",
                "fine", "tuning", "supervised", "unsupervised",
                "reinforcement", "learning", "classification",
                "regression", "clustering", "overfitting", "underfitting",
                "model", "evaluation", "pipelines", "vector", "search",
                "machine", "deep", "artificial", "intelligence",
                "ml", "ai", "dataset", "features", "labels",
                "accuracy", "precision", "recall", "f1", "score",
                "gradient", "descent", "backpropagation", "epochs",
                "batch", "loss", "function", "optimizer", "softmax",
                "attention", "mechanism", "tokenization", "tokens",
            },
            "security": {
                "vulnerability", "vulnerabilities", "cve", "cvss",
                "epss", "threat", "threats", "exploit", "exploits",
                "risk", "scoring", "firewall", "firewalls",
                "encryption", "encrypt", "authentication", "authorization",
                "penetration", "testing", "malware", "phishing",
                "zero", "day", "patch", "management",
                "incident", "response", "soc", "siem",
                "asset", "discovery", "security", "attack",
                "injection", "auth", "permission", "compliance",
                "privacy", "audit", "inspect", "trust",
                "waf", "ssl", "tls", "certificate",
                "ransomware", "botnet", "trojan", "worm",
                "ddos", "mitm", "spoofing", "breach",
                "cybersecurity", "cyber",
            },
            "science": {
                "atoms", "atom", "molecules", "molecule", "energy",
                "force", "mass", "gravity", "electricity",
                "magnetism", "chemical", "reactions", "reaction",
                "cells", "cell", "dna", "evolution", "ecosystems",
                "ecosystem", "light", "sound", "heat", "pressure",
                "motion", "space", "time", "science", "scientific",
                "experiment", "hypothesis", "theory", "law",
                "observation", "research", "physics", "chemistry",
                "biology", "matter", "element", "compound",
                "reaction", "catalyst", "thermodynamics",
            },
            "physics": {
                "force", "forces", "energy", "mass", "velocity",
                "acceleration", "momentum", "gravity", "gravitational",
                "friction", "torque", "wave", "waves", "frequency",
                "wavelength", "amplitude", "photon", "electron",
                "proton", "neutron", "quantum", "relativity",
                "newton", "einstein", "thermodynamics", "entropy",
                "kinetic", "potential", "magnetic", "electric",
                "field", "voltage", "current", "resistance",
                "circuit", "capacitor", "inductor", "ohm",
                "joule", "watt", "hertz", "nuclear", "fission",
                "fusion", "particle", "radiation",
            },
            "chemistry": {
                "atom", "atoms", "molecule", "molecules", "element",
                "elements", "compound", "compounds", "periodic",
                "table", "bond", "bonds", "ionic", "covalent",
                "reaction", "reactions", "catalyst", "catalysts",
                "oxidation", "reduction", "acid", "base", "ph",
                "solution", "solvent", "solute", "concentration",
                "mole", "molar", "stoichiometry", "organic",
                "inorganic", "polymer", "isotope", "valence",
                "electron", "shell", "orbital", "molecular",
                "chemical", "equation", "equilibrium", "kinetics",
            },
            "biology": {
                "cell", "cells", "dna", "rna", "protein", "proteins",
                "gene", "genes", "genetics", "genome", "chromosome",
                "mitosis", "meiosis", "enzyme", "enzymes",
                "membrane", "nucleus", "organelle", "tissue",
                "organ", "organism", "bacteria", "virus", "viruses",
                "fungi", "plant", "plants", "animal", "animals",
                "photosynthesis", "respiration", "metabolism",
                "evolution", "natural", "selection", "species",
                "ecosystem", "ecosystems", "ecology", "habitat",
                "food", "chain", "biodiversity", "anatomy",
                "physiology", "immunity", "hormone",
            },
            "finance": {
                "money", "currency", "currencies", "loans", "loan",
                "interest", "rate", "rates", "savings", "saving",
                "investments", "investment", "invest", "stocks",
                "stock", "bonds", "bond", "dividend", "dividends",
                "portfolio", "market", "markets", "trading", "trade",
                "capital", "assets", "asset", "liabilities",
                "liability", "equity", "revenue", "profit", "loss",
                "income", "expense", "expenses", "cash", "flow",
                "valuation", "roi", "yield", "risk", "return",
                "inflation", "deflation", "gdp", "recession",
                "compound", "principal", "amortize", "annuity",
                "portfolio", "brokerage", "exchange",
            },
            "banking": {
                "bank", "banks", "banking", "account", "accounts",
                "transaction", "transactions", "transfer", "transfers",
                "deposit", "deposits", "withdrawal", "withdrawals",
                "credit", "debit", "card", "cards", "atm",
                "checking", "savings", "mortgage", "mortgages",
                "loan", "loans", "interest", "apr", "fees",
                "overdraft", "clearing", "settlement", "swift",
                "iban", "routing", "branch", "teller", "online",
                "banking", "mobile", "payment", "payments", "check",
                "checks", "balance", "statement", "ledger",
            },
            "accounting": {
                "accounting", "bookkeeping", "ledger", "ledgers",
                "journal", "journals", "entry", "entries",
                "debit", "credit", "credits", "debits",
                "balance", "sheet", "income", "statement",
                "cash", "flow", "equity", "retained", "earnings",
                "revenue", "expense", "expenses", "payable",
                "receivable", "depreciation", "amortization",
                "accrual", "ifrs", "gaap", "audit", "auditing",
                "reconciliation", "fiscal", "tax", "taxes",
                "payroll", "invoice", "invoices", "billing",
                "assets", "liabilities", "double", "entry",
            },
            "technology": {
                "technology", "internet", "computers", "computer",
                "software", "hardware", "networks", "network",
                "networking", "data", "storage", "cloud", "servers",
                "server", "websites", "website", "browsers", "browser",
                "email", "communication", "protocol", "protocols",
                "tcp", "ip", "dns", "http", "https", "ftp",
                "router", "switch", "bandwidth", "latency",
                "operating", "system", "linux", "windows", "macos",
                "mobile", "smartphone", "tablet", "iot", "device",
                "devices", "cpu", "gpu", "ram", "disk", "ssd",
                "hdd", "usb", "bluetooth", "wifi", "ethernet",
            },
            "general_knowledge": {
                "countries", "country", "cities", "city", "weather",
                "climate", "education", "work", "society", "social",
                "history", "geography", "culture", "cultural",
                "language", "languages", "government", "politics",
                "law", "legal", "health", "medicine", "food",
                "sports", "music", "art", "literature", "film",
                "travel", "transportation", "military", "religion",
                "philosophy", "psychology", "economics", "business",
                "management", "marketing", "leadership", "career",
                "lifestyle", "entertainment", "gaming", "hobby",
            },
            "architecture": {
                "system", "design", "architecture", "component",
                "microservice", "platform", "infrastructure",
                "integration", "migrate", "migration", "upgrade",
                "transition", "structure", "pattern", "singleton",
                "factory", "observer", "strategy", "dependency",
                "injection", "distributed", "consensus", "cap",
                "theorem", "eventual", "consistency", "reverse",
                "proxy", "cdn", "load", "balancer", "message",
                "broker", "queue", "pub", "sub", "rest", "graphql",
                "grpc", "kubernetes", "docker", "container",
                "deployment", "ci", "cd", "service", "gateway",
            },
            "data": {
                "data", "database", "databases", "sql", "etl",
                "feature", "features", "dataset", "datasets",
                "statistics", "statistical", "metrics", "metric",
                "schema", "table", "tables", "query", "queries",
                "index", "indexes", "cache", "caching",
                "vector", "vectors", "tf", "idf", "cosine",
                "similarity", "clustering", "k", "means",
                "recommendation", "warehouse", "lake", "pipeline",
            },
            "general": {
                "help", "question", "what", "how", "why", "hello",
                "hi", "hey", "can", "you", "think", "recommend",
                "stuck", "ideas", "need", "assistance",
            },
        }

    def _tokens(self, text: str) -> Set[str]:
        """Tokenize text into lowercase alphanumeric words.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        return set(self._token_pattern.findall(text.lower()))

    def route(self, text: str, mode: str, intent: str = "") -> str:
        """Select the best domain.

        v3.4.0 PATCH: Added intent parameter and hard constraints to prevent
        language/grammar domains from being selected for code/explain/architect
        intents. This was the root cause of "Past Simple" appearing in
        quicksort complexity answers.

        Args:
            text: Preprocessed input.
            mode: Selected mode.
            intent: Classified intent (optional, for stricter constraints).

        Returns:
            Domain name.

        Complexity: T(n) in O(d * t) for d domains, t tokens.
        """
        tokens = self._tokens(text)
        best_domain = "general"
        best_score = 0

        # v3.4.0 PATCH: Hard constraint — for code/architect/analyze intents,
        # never allow routing to language/grammar/general domains.
        # Programming/architecture must win even with low scores.
        _technical_intents = {"code", "architect", "analyze", "explain"}
        _technical_modes = {"code", "architect", "analyze"}
        is_technical = (
            intent in _technical_intents
            or mode in _technical_modes
        )

        for domain, keywords in self._domains.items():
            if not keywords:
                continue
            # v3.4.0 PATCH: Skip non-technical domains for technical queries
            if is_technical and domain in ("general",):
                continue
            # Use intersection count (not Jaccard) — more appropriate
            # because input texts have many tokens but domain sets
            # are curated keyword lists.
            score = len(tokens & keywords)
            if score > best_score:
                best_score = score
                best_domain = domain

        # Mode-based fallback: if no domain matched but mode is code/analyze,
        # default to programming domain.
        if mode in {"code", "analyze"} and best_domain == "general":
            best_domain = "programming"
        # If mode is architect, default to architecture domain.
        if mode == "architect" and best_domain == "general":
            best_domain = "architecture"
        # Identity intent: force domain to "condrox" so self-knowledge items
        # are retrieved instead of generic corpus content
        if intent == "identity":
            best_domain = "condrox"
        # v3.4.0 PATCH: If intent is explain and domain is still general,
        # check for technical signals in the text before falling back.
        if intent == "explain" and best_domain == "general":
            _tech_signals = {
                "quicksort", "mergesort", "bubble", "complexity", "algorithm",
                "recursion", "recursive", "function", "python", "javascript",
                "sort", "search", "tree", "graph", "stack", "queue",
            }
            if tokens & _tech_signals:
                best_domain = "programming"

        return best_domain

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run domain routing with KnowledgeRouterV2 enrichment.

        Uses the built-in keyword router first, then enriches with
        KnowledgeRouterV2 if available for fallback domain suggestion.

        v3.4.0 PATCH: Now passes intent to route() for stricter constraints.

        Complexity: T(n) ∈ O(d * n + d2 * k), S(n) ∈ O(n).
        """
        domain = self.route(state.processed_input, state.mode, state.intent)

        # Enrich with KnowledgeRouterV2 for fallback domain
        context = dict(state.context)
        try:
            v2_decision = controller.knowledge_router_v2.route(state.processed_input)
            if v2_decision.confidence > 0.5:
                # v3.4.0 PATCH: Don't let KnowledgeRouterV2 override a technical
                # domain with a non-technical one for code/architect/analyze intents
                _technical_intents = {"code", "architect", "analyze"}
                _technical_domains = {
                    "programming", "architecture", "security", "data",
                    "ai_ml", "science", "physics",
                    "chemistry", "biology", "finance", "banking",
                    "accounting", "technology",
                }
                if state.intent in _technical_intents and domain in _technical_domains:
                    # Keep the technical domain, ignore V2's non-technical suggestion
                    if v2_decision.domain not in _technical_domains:
                        pass  # Keep original domain
                    else:
                        domain = v2_decision.domain
                else:
                    domain = v2_decision.domain
            context["fallback_domain"] = v2_decision.fallback_domain
            context["domain_confidence"] = v2_decision.confidence
        except Exception:
            pass

        return state.with_field(domain=domain, context=context)
