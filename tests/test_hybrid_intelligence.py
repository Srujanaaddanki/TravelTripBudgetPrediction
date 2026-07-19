import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pages.plan_trip import get_ml_proxy_destination, _resolve_destination, format_budget_display
from src.services.confidence_engine import ConfidenceEngine
from src.services.travel_intelligence import TravelIntelligenceEngine


class TestHybridDestinationIntelligence(unittest.TestCase):
    def test_ml_proxy_destination_mappings(self):
        """Test proxy mapping helper handles dynamic profiling and scoring rules."""
        encoder_classes = ["tirupati", "dubai", "spiti", "manali", "delhi"]
        
        # Annavaram (temple category, low altitude, India) should match tirupati (temple/tropical/medium pop)
        proxy, score = get_ml_proxy_destination(
            "Annavaram", 
            encoder_classes=encoder_classes,
            country="India", 
            tourism_category="Temple", 
            altitude=100, 
            weather_profile="tropical"
        )
        self.assertEqual(proxy, "tirupati")
        self.assertTrue(score > 70)
        
        # Paris (metro category, international/france) should match dubai (metro category)
        proxy, score = get_ml_proxy_destination(
            "Paris", 
            encoder_classes=encoder_classes,
            country="France", 
            tourism_category="Metropolitan City"
        )
        self.assertEqual(proxy, "dubai")

        # High altitude Himalayan Hill station should match manali/shimla/spiti depending on altitude
        proxy, score = get_ml_proxy_destination(
            "Some peak", 
            encoder_classes=encoder_classes,
            country="India", 
            tourism_category="Hill Station", 
            altitude=3500, 
            weather_profile="himalayan"
        )
        self.assertEqual(proxy, "spiti")

    @patch("src.pages.plan_trip._geo_service")
    @patch("src.pages.plan_trip._dest_cache")
    def test_resolve_destination_pipeline(self, mock_cache, mock_geo):
        """Test the multi-step destination resolution pipeline."""
        mock_cache.get_intelligence_cache.return_value = None
        mock_cache.get_cached.return_value = None
        
        # Test Exact match
        encoder_classes = ["Goa", "Tirupati", "Delhi", "Manali", "Kukunate"]
        res = _resolve_destination("Goa", encoder_classes)
        self.assertTrue(res["is_known"])
        self.assertEqual(res["matched_dest"], "goa")
        self.assertEqual(res["match_tier"], "exact")

        # Test Geoapify geocoding success (Unknown destination)
        mock_geo.validate_destination.return_value = {
            "valid": True,
            "lat": 17.06,
            "lng": 82.40,
            "country": "India",
            "state": "Andhra Pradesh",
            "display_name": "Annavaram, Andhra Pradesh"
        }
        
        # Mock Gemini metadata classifier response
        with patch("src.services.gemini_service.GeminiService.resolve_unknown_destination_metadata") as mock_meta:
            mock_meta.return_value = {
                "country": "India",
                "state": "Andhra Pradesh",
                "altitude": 100.0,
                "tourism_category": "Temple",
                "weather_profile": "tropical",
                "population_profile": "medium"
            }
            
            res_unknown = _resolve_destination("Annavaram", encoder_classes)
            self.assertFalse(res_unknown["is_known"])
            self.assertEqual(res_unknown["matched_dest"], "tirupati")
            self.assertEqual(res_unknown["match_tier"], "geo_validated")
            self.assertEqual(res_unknown["dst_coords"], (17.06, 82.40))
            self.assertTrue("proxy_score" in res_unknown)

        # Test completely unresolved destination (no coordinates, no country, no API match)
        mock_geo.validate_destination.return_value = {"valid": False}
        with patch("src.services.gemini_service.GeminiService.suggest_alternative_destination") as mock_suggest:
            mock_suggest.return_value = {"valid": False}
            res_unresolved = _resolve_destination("xyzqweasd", encoder_classes)
            self.assertEqual(res_unresolved["match_tier"], "unresolved_error")
            self.assertIsNone(res_unresolved["matched_dest"])
            self.assertIsNone(res_unresolved["dst_coords"])

    def test_confidence_engine_scaling(self):
        """Verify confidence scores match scaled requirements for known, API only, AI and fallback resolutions."""
        engine = ConfidenceEngine()
        
        # Known destination: 95-100%
        res_known = engine.calculate_confidence(
            dataset_insights={"has_data": True, "similar_count": 6},
            mode_comparison={"distance_km": 500},
            ml_prediction=15000,
            is_known=True,
            resolution_type="known"
        )
        self.assertEqual(res_known["score"], 100)
        self.assertEqual(res_known["level"], "Dataset Verified")

        # API estimated: 65-80%
        res_api = engine.calculate_confidence(
            dataset_insights={"has_data": False, "similar_count": 0},
            mode_comparison={"distance_km": 500},
            ml_prediction=15000,
            is_known=False,
            resolution_type="api_only"
        )
        self.assertEqual(res_api["score"], 80)
        self.assertEqual(res_api["level"], "API Estimation")

        # AI approximation (Gemini): 50-70%
        res_gemini = engine.calculate_confidence(
            dataset_insights={"has_data": False, "similar_count": 0},
            mode_comparison={"distance_km": 500},
            ml_prediction=15000,
            is_known=False,
            resolution_type="gemini_approx"
        )
        self.assertEqual(res_gemini["score"], 70)
        self.assertEqual(res_gemini["level"], "AI Approximation")

        # Fallback: <50%
        res_fallback = engine.calculate_confidence(
            dataset_insights={"has_data": False, "similar_count": 0},
            mode_comparison={"distance_km": 0},
            ml_prediction=15000,
            is_known=False,
            resolution_type="failed"
        )
        self.assertTrue(res_fallback["score"] < 50)
        self.assertEqual(res_fallback["level"], "Low Confidence")

    def test_budget_range_display(self):
        """Test that budget ranges are formatted correctly based on confidence scores."""
        # 70%+ -> Exact budget
        self.assertEqual(format_budget_display(10000, 75), "Rs.10,000")
        
        # 50-70% -> ±15%
        self.assertEqual(format_budget_display(10000, 60), "Rs.8,500 - Rs.11,500")

        # 30-50% -> ±25%
        self.assertEqual(format_budget_display(10000, 40), "Rs.7,500 - Rs.12,500")

        # <30% -> Very Low Confidence Estimation
        self.assertEqual(format_budget_display(10000, 20), "Very Low Confidence Estimation")


if __name__ == "__main__":
    unittest.main()
