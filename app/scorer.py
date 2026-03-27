from __future__ import annotations


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _label_from_score(score: int) -> str:
    if score >= 80:
        return "Highly Suspicious / Likely Malware"
    if score >= 60:
        return "Suspicious"
    if score >= 40:
        return "Needs Review"
    return "Low Suspicion"


def compute_pe_risk(pe_info: dict) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    is_pe = pe_info.get("is_pe", False)
    avg_entropy = float(pe_info.get("avg_section_entropy", 0.0))
    max_entropy = float(pe_info.get("max_section_entropy", 0.0))
    imports_count = int(pe_info.get("imports_count", 0))
    suspicious_names = pe_info.get("suspicious_section_names", []) or []
    num_sections = int(pe_info.get("num_sections", 0))
    overlay_ratio = float(pe_info.get("overlay_ratio", 0.0))
    exec_write_sections = pe_info.get("executable_writable_sections", []) or []
    high_entropy_sections = pe_info.get("high_entropy_sections", []) or []

    if not is_pe:
        score += 45
        reasons.append("File could not be parsed cleanly as a normal PE executable.")

    if avg_entropy >= 7.2:
        score += 20
        reasons.append("Average section entropy is high.")
    elif avg_entropy >= 6.8:
        score += 10
        reasons.append("Average section entropy is mildly elevated.")

    if max_entropy >= 7.6:
        score += 18
        reasons.append("At least one section has very high entropy.")
    elif max_entropy >= 7.2:
        score += 10
        reasons.append("At least one section has high entropy.")

    if high_entropy_sections:
        reasons.append(f"High-entropy sections: {', '.join(high_entropy_sections[:4])}.")

    if imports_count == 0:
        score += 18
        reasons.append("No imports were found.")
    elif imports_count <= 5:
        score += 12
        reasons.append("Very low import count may indicate packing or manual API resolution.")
    elif imports_count <= 15:
        score += 6
        reasons.append("Low import count is mildly suspicious.")

    if suspicious_names:
        score += 22
        reasons.append(f"Suspicious section names found: {', '.join(suspicious_names)}.")

    if exec_write_sections:
        score += 20
        reasons.append(
            f"Executable and writable sections found: {', '.join(exec_write_sections)}."
        )

    if overlay_ratio >= 0.25:
        score += 18
        reasons.append("Large overlay appended to the PE file.")
    elif overlay_ratio >= 0.10:
        score += 10
        reasons.append("Noticeable overlay data was found.")

    if num_sections <= 2:
        score += 8
        reasons.append("Very small number of sections can be suspicious.")
    elif num_sections >= 10:
        score += 6
        reasons.append("Unusually large number of sections may indicate tampering or packing.")

    return _clamp_score(score), reasons


def compute_suspicion_score(pe_info: dict, cnn_info: dict | None = None) -> dict:
    pe_score, reasons = compute_pe_risk(pe_info)

    cnn_available = bool(
        cnn_info
        and cnn_info.get("available")
        and cnn_info.get("malware_specific")
        and cnn_info.get("malware_score") is not None
    )

    cnn_score = int(cnn_info.get("malware_score")) if cnn_available else None
    cnn_probability = float(cnn_info.get("malware_probability")) if cnn_available else None

    cnn_weight = 0.70
    pe_weight = 0.30
    cnn_used = cnn_available
    cnn_bonus = 0

    if cnn_available:
        blended_score = _clamp_score((cnn_weight * cnn_score) + (pe_weight * pe_score))

        if cnn_probability >= 0.95:
            final_score = max(blended_score, 85, cnn_score)
            reasons.insert(
                0,
                f"Malware-trained CNN predicted very high malware probability ({cnn_probability:.1%}), so CNN output dominates the verdict.",
            )
        elif cnn_probability >= 0.85:
            final_score = max(blended_score, 75)
            reasons.insert(
                0,
                f"Malware-trained CNN predicted high malware probability ({cnn_probability:.1%}).",
            )
        elif cnn_probability >= 0.70:
            final_score = blended_score
            reasons.insert(
                0,
                f"Malware-trained CNN predicted suspicious malware probability ({cnn_probability:.1%}).",
            )
        elif cnn_probability <= 0.15:
            final_score = min(blended_score, max(10, pe_score))
            reasons.insert(
                0,
                f"Malware-trained CNN predicted low malware probability ({cnn_probability:.1%}).",
            )
        else:
            final_score = blended_score
            reasons.insert(
                0,
                f"Malware-trained CNN probability was {cnn_probability:.1%}; PE indicators were used as secondary support.",
            )

        if pe_score >= 80 and final_score < 55:
            final_score = 55
            reasons.append(
                "PE structure is still highly suspicious even though the CNN score is lower."
            )

        for item in (cnn_info.get("reasons") or [])[:3]:
            reasons.append(f"CNN: {item}")

        cnn_bonus = max(0, final_score - pe_score)

    else:
        final_score = pe_score
        if cnn_info and cnn_info.get("status"):
            reasons.insert(
                0,
                f"Malware CNN was unavailable ({cnn_info.get('status')}); verdict falls back to PE analysis."
            )

    final_score = _clamp_score(final_score)
    label = _label_from_score(final_score)

    return {
        "score": final_score,
        "label": label,
        "reasons": reasons,
        "rule_score": pe_score,                 # keep old key for UI compatibility
        "cnn_used": cnn_used,
        "cnn_visual_score": cnn_score,         # keep old key for UI compatibility
        "cnn_bonus": cnn_bonus,                # keep old key for UI compatibility
        "cnn_probability": round(cnn_probability, 4) if cnn_available else None,
        "cnn_weight": cnn_weight,
        "pe_weight": pe_weight,
        "blend_mode": "cnn_primary",
    }
