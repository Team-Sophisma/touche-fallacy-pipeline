class FallacyArchitectureKnowledgeBase:
    _profiles = {
        "authority": {
            "display_name": "Appeal to Authority",
            "definition": "Treats a claim as true because an authority, expert, institution, or public figure is invoked.",
            "claim_pattern": "A conclusion is presented as acceptable because a named or implied source supports it.",
            "support_pattern": "The support depends on status, credentials, office, reputation, or institutional identity.",
            "hidden_reasoning_bridge": "If this authority says or implies it, the claim should be accepted.",
            "failure_point": "Authority is substituted for sufficient evidence or domain-relevant reasoning.",
            "valid_lookalike": "A valid argument cites a relevant authority together with evidence, limits, or verifiable grounds.",
            "common_cues": ["expert", "doctor", "scientist", "official", "organization", "said"],
            "confusion_risks": ["institutional evidence", "political authority", "celebrity endorsement"],
        },
        "black-white": {
            "display_name": "Black-White Fallacy",
            "definition": "Frames a situation as having only two options when additional alternatives or degrees are plausible.",
            "claim_pattern": "The conclusion forces a binary choice between two extremes.",
            "support_pattern": "The support excludes middle positions, tradeoffs, or alternative explanations.",
            "hidden_reasoning_bridge": "If one option is rejected, the opposite extreme must be accepted.",
            "failure_point": "The argument collapses a multi-option problem into a false dichotomy.",
            "valid_lookalike": "A valid binary argument shows that the options are genuinely exhaustive in the given context.",
            "common_cues": ["either", "or", "only", "must", "never", "always"],
            "confusion_risks": ["strict policy choices", "clear logical opposites"],
        },
        "hasty_generalization": {
            "display_name": "Hasty Generalization",
            "definition": "Draws a broad conclusion from insufficient, unrepresentative, or anecdotal evidence.",
            "claim_pattern": "A general rule is inferred from limited observations.",
            "support_pattern": "The support relies on one case, a small sample, or an isolated experience.",
            "hidden_reasoning_bridge": "This small set of examples is enough to characterize the whole class.",
            "failure_point": "The sample does not justify the scope of the conclusion.",
            "valid_lookalike": "A valid generalization uses representative data and states an appropriately limited claim.",
            "common_cues": ["everyone", "always", "never", "all", "people", "once"],
            "confusion_risks": ["population claims", "empirical summaries"],
        },
        "natural": {
            "display_name": "Appeal to Nature",
            "definition": "Claims something is better, right, or justified because it is natural or inherent.",
            "claim_pattern": "A normative conclusion is tied to what is described as natural.",
            "support_pattern": "The support invokes nature, biology, purity, or inherent properties.",
            "hidden_reasoning_bridge": "Natural properties are automatically good, correct, or preferable.",
            "failure_point": "Naturalness is treated as sufficient proof of value or correctness.",
            "valid_lookalike": "A valid argument explains why a natural property is causally or ethically relevant.",
            "common_cues": ["natural", "nature", "biological", "inherent", "pure", "organic"],
            "confusion_risks": ["tradition", "health claims", "biological explanation"],
            "notes": [
                "The paper identifies natural vs. tradition as a recurring semantic confusion pair.",
            ],
        },
        "population": {
            "display_name": "Appeal to Population",
            "definition": "Treats a claim as true or acceptable because many people believe or do it.",
            "claim_pattern": "A conclusion is justified by popularity or majority acceptance.",
            "support_pattern": "The support emphasizes crowds, public opinion, trends, or common practice.",
            "hidden_reasoning_bridge": "If many people accept it, it must be true or right.",
            "failure_point": "Popularity is substituted for evidence or logical support.",
            "valid_lookalike": "A valid argument uses popularity only when popularity itself is the relevant evidence.",
            "common_cues": ["everyone", "most people", "popular", "majority", "society", "common"],
            "confusion_risks": ["tradition", "survey evidence", "social norms"],
        },
        "slippery_slope": {
            "display_name": "Slippery Slope",
            "definition": "Claims an action will trigger a chain of worsening outcomes without adequate causal support.",
            "claim_pattern": "A conclusion warns that one step inevitably leads to severe consequences.",
            "support_pattern": "The support links events through an escalating causal chain.",
            "hidden_reasoning_bridge": "Allowing the first step makes later harmful steps unavoidable.",
            "failure_point": "The causal chain is asserted rather than demonstrated.",
            "valid_lookalike": "A valid causal warning provides evidence for each step and avoids inevitability claims.",
            "common_cues": ["lead to", "next", "eventually", "before long", "inevitable", "open the door"],
            "confusion_risks": ["risk analysis", "worse problems"],
        },
        "tradition": {
            "display_name": "Appeal to Tradition",
            "definition": "Claims something is right or better because it has long been believed or practiced.",
            "claim_pattern": "A conclusion is justified by historical continuity or long-standing practice.",
            "support_pattern": "The support invokes custom, heritage, history, or established ways.",
            "hidden_reasoning_bridge": "Long-standing use or belief is sufficient proof of correctness.",
            "failure_point": "Historical persistence is substituted for present evidence or reasoning.",
            "valid_lookalike": "A valid argument explains why continuity, precedent, or inherited practice is relevant.",
            "common_cues": ["tradition", "always", "for generations", "historically", "custom", "heritage"],
            "confusion_risks": ["natural", "population", "precedent-based reasoning"],
            "notes": [
                "The paper reports confusion where tradition-based arguments can be mislabeled as natural.",
            ],
        },
        "worse_problems": {
            "display_name": "Worse Problems",
            "definition": "Dismisses a concern by pointing to another issue presented as more serious.",
            "claim_pattern": "The target issue is rejected or minimized because a worse issue exists.",
            "support_pattern": "The support compares harms and redirects attention to a larger problem.",
            "hidden_reasoning_bridge": "A problem matters only if no worse problem can be named.",
            "failure_point": "Relative severity is used to avoid addressing the original claim.",
            "valid_lookalike": "A valid prioritization argument compares constraints, urgency, and available resources.",
            "common_cues": ["worse", "bigger problem", "instead", "more important", "why care", "real issue"],
            "confusion_risks": ["policy prioritization", "slippery slope", "resource allocation"],
        },
    }

    def describe(self, fallacy_type: str) -> dict:
        normalized = self._normalize(fallacy_type)
        profile = self._profiles.get(normalized)

        if profile is None:
            display_name = normalized.replace("_", " ").replace("-", " ").title()
            return {
                "fallacy_type": normalized,
                "display_name": display_name,
                "definition": "",
                "claim_pattern": "",
                "support_pattern": "",
                "hidden_reasoning_bridge": "",
                "failure_point": "",
                "valid_lookalike": "",
                "common_cues": [],
                "confusion_risks": [],
                "notes": [],
            }

        return {
            "fallacy_type": normalized,
            **profile,
        }

    def _normalize(self, fallacy_type: str) -> str:
        normalized = str(fallacy_type or "").strip().lower()
        if normalized == "blackwhite":
            return "black-white"
        return normalized
