def compute_suspicion_score(pe_info: dict, cnn_info: dict | None = None) -> dict:
    rule_score = 0
    reasons = []

    is_pe = pe_info.get("is_pe", False)
    avg_entropy = pe_info.get("avg_section_entropy", 0.0)
    imports_count = pe_info.get("imports_count", 0)
    suspicious_names = pe_info.get("suspicious_section_names", [])
    num_sections = pe_info.get("num_sections", 0)

    if not is_pe:
        rule_score += 30
        reasons.append("File could not be parsed as a normal PE executable.")

    if avg_entropy >= 7.4:
        rule_score += 30
        reasons.append("Very high average section entropy may suggest packing or strong obfuscation.")
    elif avg_entropy >= 7.0:
        rule_score += 20
        reasons.append("High section entropy may indicate compression or unusual structure.")
    elif avg_entropy >= 6.8:
        rule_score += 10
        reasons.append("Slightly elevated entropy was observed, but this alone is not a strong malware signal.")

    if imports_count == 0:
        rule_score += 25
        reasons.append("No imports were found, which is often seen in packed or manually resolved binaries.")
    elif imports_count <= 5:
        rule_score += 18
        reasons.append("Very low import count may indicate a packed or minimized binary.")
    elif imports_count <= 15:
        rule_score += 8
        reasons.append("Low import count is mildly suspicious.")

    if suspicious_names:
        rule_score += 25
        reasons.append(f"Suspicious section names found: {', '.join(suspicious_names)}.")

    if num_sections <= 2:
        rule_score += 10
        reasons.append("Very small number of sections can be suspicious.")

    rule_score = min(rule_score, 100)

    cnn_available = bool(cnn_info and cnn_info.get("available") and cnn_info.get("visual_score") is not None)
    cnn_visual_score = int(cnn_info["visual_score"]) if cnn_available else None
    cnn_bonus = 0
    cnn_used = False

    strong_pe_signal = (
        rule_score >= 25
        or bool(suspicious_names)
        or avg_entropy >= 7.0
        or imports_count <= 5
        or not is_pe
    )

    if cnn_available:
        strong_signal_count = int(cnn_info.get("strong_signal_count", 0))

        strong_visual = cnn_visual_score >= 75 and strong_signal_count >= 2
        moderate_visual = cnn_visual_score >= 65 and strong_signal_count >= 2

        if strong_pe_signal and strong_visual:
            cnn_bonus = min(12, max(6, (cnn_visual_score - 60) // 3))
            cnn_used = True
            reasons.append(
                f"Pretrained CNN texture analysis supported the PE findings and added a limited bonus of +{cnn_bonus}."
            )
            for item in cnn_info.get("reasons", [])[:2]:
                reasons.append(f"CNN: {item}")
        elif strong_pe_signal and moderate_visual:
            cnn_bonus = min(6, max(3, (cnn_visual_score - 60) // 5))
            cnn_used = True
            reasons.append(
                f"Pretrained CNN texture analysis slightly supported the PE findings and added +{cnn_bonus}."
            )
            for item in cnn_info.get("reasons", [])[:1]:
                reasons.append(f"CNN: {item}")
        else:
            reasons.append(
                "CNN output was treated as informational only because the PE indicators were not strong enough."
            )
    else:
        if cnn_info and cnn_info.get("status"):
            reasons.append(f"Pretrained CNN could not be used: {cnn_info['status']}.")

    final_score = min(100, rule_score + cnn_bonus)

    if final_score >= 70:
        label = "Highly Suspicious / Possibly Packed"
    elif final_score >= 40:
        label = "Moderately Suspicious"
    else:
        label = "Low Suspicion"

    return {
        "score": final_score,
        "label": label,
        "reasons": reasons,
        "rule_score": rule_score,
        "cnn_used": cnn_used,
        "cnn_visual_score": cnn_visual_score,
        "cnn_bonus": cnn_bonus,
    }
