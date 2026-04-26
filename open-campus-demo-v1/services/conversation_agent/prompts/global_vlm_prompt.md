# Global VLM Prompt

目的:
- 全体画像から場の状態を推定する
- 個人特定はしない

出力はJSONのみ:
{
  "scene_summary": "...",
  "people_flow": "increasing|stable|decreasing|unknown",
  "attention_target": "danya|nanamaru|display|passing|unknown",
  "crowd_state": "none|single|small_group|crowd|unknown",
  "movement_state": "passing|stopped|approaching|leaving|mixed|unknown",
  "engagement_estimate": 0.0,
  "confusion_estimate": 0.0,
  "leaving_risk": 0.0,
  "notable_event": "...",
  "recommended_interaction": "hook|intro|deepen|quiz|crowd|closing|idle"
}

