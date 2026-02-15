#!/usr/bin/env python3
"""
Improved Ensemble Model with Better Rules
"""

from sentence_transformers import SentenceTransformer
import pickle
import json
import re


class ResumeAnalyzer:
    """Complete resume analysis system"""

    def __init__(self, model_path="models"):
        # Load ML model
        self.embedder = SentenceTransformer(f"{model_path}/sentence_transformer")

        with open(f"{model_path}/classifier.pkl", "rb") as f:
            self.classifier = pickle.load(f)

        with open(f"{model_path}/label_map.json", "r") as f:
            self.label_map = json.load(f)

        self.id_to_label = {v: k for k, v in self.label_map.items()}

        # ✅ BALANCED ENSEMBLE
        self.ml_weight = 0.5
        self.rule_weight = 0.5

    def extract_features(self, text):
        """Extract structured features from resume"""

        return {
            "cgpa": self._extract_cgpa(text),
            "skills": self._extract_skills(text),
            "has_internship": self._has_keyword(text, ["intern", "internship"]),
            "has_projects": self._has_keyword(text, ["project"]),
            "has_research": self._has_keyword(
                text, ["research", "publication", "paper"]
            ),
            "has_certifications": self._has_keyword(
                text, ["certified", "certificate", "certification"]
            ),
        }

    def _extract_cgpa(self, text):
        """Extract CGPA from text"""
        patterns = [
            r"CGPA[:\s]*(\d+\.?\d*)",
            r"GPA[:\s]*(\d+\.?\d*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                cgpa = float(match.group(1))
                return cgpa if cgpa <= 10 else cgpa / 10
        return None

    def _extract_skills(self, text):
        """Extract technical skills"""
        skills_db = [
            "Python",
            "Java",
            "JavaScript",
            "C++",
            "C",
            "React",
            "Node.js",
            "Angular",
            "Vue",
            "Machine Learning",
            "Deep Learning",
            "TensorFlow",
            "PyTorch",
            "Keras",
            "MongoDB",
            "MySQL",
            "PostgreSQL",
            "Redis",
            "AWS",
            "Azure",
            "Docker",
            "Kubernetes",
            "Git",
        ]

        text_lower = text.lower()
        return [s for s in skills_db if s.lower() in text_lower]

    def _has_keyword(self, text, keywords):
        """Check if text contains any keyword"""
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    def rule_based_score(self, features):
        """Generate rule-based scores for each category - IMPROVED VERSION"""

        cgpa = features["cgpa"] or 0
        skills_count = len(features["skills"])

        scores = {
            "Private Job": 0.0,
            "Higher Studies": 0.0,
            "Research Field": 0.0,
            "Skill Improvement": 0.0,
        }

        # === PRIVATE JOB SCORING ===
        if features["has_internship"]:
            scores["Private Job"] += 35
        if skills_count >= 10:
            scores["Private Job"] += 30
        elif skills_count >= 8:
            scores["Private Job"] += 20
        if cgpa >= 7.5:
            scores["Private Job"] += 20
        if features["has_projects"]:
            scores["Private Job"] += 15

        # === HIGHER STUDIES SCORING ===
        # CGPA is the strongest signal
        if cgpa >= 9.0:
            scores["Higher Studies"] += 50  # ✅ Strong signal for your case
        elif cgpa >= 8.5:
            scores["Higher Studies"] += 40
        elif cgpa >= 8.0:
            scores["Higher Studies"] += 25

        if features["has_projects"]:
            scores["Higher Studies"] += 15
        if skills_count >= 6:
            scores["Higher Studies"] += 10

        # Slight penalty if has internship (more likely Private Job)
        if features["has_internship"]:
            scores["Higher Studies"] -= 5

        # === RESEARCH FIELD SCORING ===
        if features["has_research"]:
            scores["Research Field"] += 50
        if cgpa >= 9.0:
            scores["Research Field"] += 35
        elif cgpa >= 8.5:
            scores["Research Field"] += 25
        if features["has_projects"]:
            scores["Research Field"] += 10

        # === SKILL IMPROVEMENT SCORING ===
        if skills_count < 4:
            scores["Skill Improvement"] += 40
        elif skills_count < 6:
            scores["Skill Improvement"] += 20

        if cgpa < 6.0:
            scores["Skill Improvement"] += 35
        elif cgpa < 7.0:
            scores["Skill Improvement"] += 20

        if not features["has_internship"]:
            scores["Skill Improvement"] += 15
        if not features["has_projects"]:
            scores["Skill Improvement"] += 15

        # Ensure no negative scores
        scores = {k: max(0, v) for k, v in scores.items()}

        # Normalize to 0-1
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        else:
            scores = {k: 0.25 for k in scores.keys()}

        return scores

    def predict(self, resume_text):
        """Make ensemble prediction"""

        # Extract features
        features = self.extract_features(resume_text)

        # ML prediction
        embedding = self.embedder.encode([resume_text])
        ml_pred_id = self.classifier.predict(embedding)[0]
        ml_probs = self.classifier.predict_proba(embedding)[0]

        ml_prediction = self.id_to_label[ml_pred_id]
        ml_probs_dict = {self.id_to_label[i]: prob for i, prob in enumerate(ml_probs)}

        # Rule-based prediction
        rule_probs = self.rule_based_score(features)
        rule_prediction = max(rule_probs, key=rule_probs.get)

        # Ensemble: combine ML and rules
        ensemble_probs = {}
        for category in self.label_map.keys():
            ensemble_probs[category] = self.ml_weight * ml_probs_dict.get(
                category, 0
            ) + self.rule_weight * rule_probs.get(category, 0)

        # Normalize probabilities
        total = sum(ensemble_probs.values())
        if total > 0:
            ensemble_probs = {k: v / total for k, v in ensemble_probs.items()}

        final_category = max(ensemble_probs, key=ensemble_probs.get)
        confidence = ensemble_probs[final_category]

        return {
            "category": final_category,
            "confidence": confidence,
            "probabilities": ensemble_probs,
            "features": features,
            "ml_prediction": ml_prediction,
            "rule_prediction": rule_prediction,
            "ml_probabilities": ml_probs_dict,
            "rule_probabilities": rule_probs,
        }

    def generate_recommendations(self, result):
        """Generate actionable recommendations"""

        category = result["category"]
        features = result["features"]
        recommendations = []

        if category == "Private Job":
            if len(features["skills"]) < 10:
                recommendations.append("Learn 2-3 more in-demand technologies")
            if not features["has_certifications"]:
                recommendations.append("Get industry certifications (AWS, Azure, etc.)")
            recommendations.append("Practice DSA on LeetCode/HackerRank")
            recommendations.append("Build strong GitHub portfolio")

        elif category == "Higher Studies":
            recommendations.append("Prepare for GRE/GATE exams")
            recommendations.append("Research universities and programs")
            recommendations.append("Request letters of recommendation")
            recommendations.append("Write compelling Statement of Purpose")

        elif category == "Research Field":
            recommendations.append("Publish papers in conferences/journals")
            recommendations.append("Connect with professors in your field")
            recommendations.append("Apply for research internships")
            recommendations.append("Consider PhD programs")

        else:  # Skill Improvement
            if len(features["skills"]) < 5:
                recommendations.append(
                    "Learn core programming: Python, Java, or JavaScript"
                )
            if not features["has_projects"]:
                recommendations.append("Build 3-4 substantial projects")
            recommendations.append("Take online courses (Coursera, Udemy)")
            recommendations.append("Participate in hackathons")
            if features["cgpa"] and features["cgpa"] < 7.0:
                recommendations.append("Focus on improving academic performance")

        return recommendations


# Test
if __name__ == "__main__":
    import pandas as pd

    print("Loading analyzer...")
    analyzer = ResumeAnalyzer()

    print("Loading test data...")
    test_df = pd.read_csv("data/test.csv")

    # Test on first resume
    sample = test_df.iloc[0]

    print("\n" + "=" * 70)
    print("ENSEMBLE PREDICTION TEST")
    print("=" * 70)

    print(f"\nTrue Category: {sample['category']}")

    result = analyzer.predict(sample["text"])

    print(f"\nFinal Prediction: {result['category']}")
    print(f"Confidence: {result['confidence']:.1%}")

    print(f"\n🤖 ML Model says: {result['ml_prediction']}")
    print(f"📏 Rules say: {result['rule_prediction']}")

    print(f"\n🎯 Ensemble Probabilities (50% ML + 50% Rules):")
    for cat, prob in sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(prob * 50)
        print(f"  {cat:20} {prob:5.1%} {bar}")

    # Show breakdown
    print(f"\n📊 Breakdown:")
    print(f"  ML Probabilities (50% weight):")
    for cat, prob in sorted(
        result["ml_probabilities"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"    {cat:20} {prob:5.1%}")

    print(f"\n  Rule Probabilities (50% weight):")
    for cat, prob in sorted(
        result["rule_probabilities"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"    {cat:20} {prob:5.1%}")

    print(f"\n🔍 Extracted Features:")
    print(f"  CGPA: {result['features']['cgpa']}")
    print(
        f"  Skills: {len(result['features']['skills'])} - {', '.join(result['features']['skills'])}"
    )
    print(f"  Has Internship: {'✓' if result['features']['has_internship'] else '✗'}")
    print(f"  Has Projects: {'✓' if result['features']['has_projects'] else '✗'}")
    print(f"  Has Research: {'✓' if result['features']['has_research'] else '✗'}")

    print(f"\n💡 Recommendations:")
    recommendations = analyzer.generate_recommendations(result)
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
