def build_explanation(
    pe_info: dict,
    score_info: dict,
    image_info: dict,
    cnn_info: dict | None = None,
) -> str:
    lines = []

    lines.append(
        f"The uploaded file was converted into a grayscale image of size "
        f"{image_info['width']} x {image_info['height']}."
    )

    if image_info.get("sampled_for_image"):
        lines.append(
            f"The file was larger than the image budget, so bytes were sampled with stride "
            f"{image_info.get('sampling_stride', 1)} to build the image efficiently."
        )

    if pe_info.get("is_pe"):
        lines.append(
            f"The file appears to be a valid PE executable with "
            f"{pe_info.get('num_sections', 0)} sections and "
            f"{pe_info.get('imports_count', 0)} imported functions."
        )
        lines.append(
            f"The average section entropy is {pe_info.get('avg_section_entropy', 0.0):.2f}."
        )
    else:
        lines.append("The file could not be fully parsed as a standard PE executable.")

    if cnn_info and cnn_info.get("available"):
        lines.append(
            f"A pretrained CNN backbone ({cnn_info.get('model_name', 'CNN')}) also inspected the grayscale byte image. "
            f"It produced a visual anomaly score of {cnn_info.get('visual_score', 0)}/100."
        )
        lines.append(
            f"This CNN is used only as a supporting visual signal, not as a final malware verdict."
        )
    elif cnn_info:
        lines.append(
            f"Pretrained CNN analysis was skipped because: {cnn_info.get('error') or cnn_info.get('status', 'unknown reason')}."
        )

    lines.append(
        f"Based on the combined analysis, the file received a suspicion score of "
        f"{score_info['score']}/100 and is classified as: {score_info['label']}."
    )

    if score_info.get("cnn_used"):
        lines.append(
            f"The final score blends the PE rule score ({score_info.get('rule_score', 0)}/100) "
            f"with the CNN visual score ({score_info.get('cnn_visual_score', 0)}/100)."
        )

    if score_info["reasons"]:
        lines.append("Main reasons:")
        for reason in score_info["reasons"]:
            lines.append(f"- {reason}")

    lines.append(
        "This is a hackathon MVP and not a full antivirus engine. "
        "It gives an explainable early warning based on PE structure, entropy, and pretrained CNN visual analysis."
    )

    return "\n".join(lines)
