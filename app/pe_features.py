from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pefile


KNOWN_SUSPICIOUS_SECTION_NAMES = {
    ".upx",
    "upx0",
    "upx1",
    "upx2",
    ".aspack",
    ".adata",
    ".packed",
    "pec1",
    "pec2",
    ".petite",
    ".themida",
    ".vmp0",
    ".vmp1",
}


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0

    counts = Counter(data)
    total = len(data)
    entropy = 0.0

    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def _safe_section_name(section) -> str:
    return section.Name.decode(errors="ignore").strip("\x00").strip()


def _section_flags(section) -> dict:
    chars = int(section.Characteristics)
    return {
        "readable": bool(chars & 0x40000000),
        "writable": bool(chars & 0x80000000),
        "executable": bool(chars & 0x20000000),
    }


def extract_pe_features(file_path: str) -> dict:
    file_size = Path(file_path).stat().st_size

    result = {
        "is_pe": False,
        "num_sections": 0,
        "section_names": [],
        "section_entropies": [],
        "avg_section_entropy": 0.0,
        "max_section_entropy": 0.0,
        "high_entropy_sections": [],
        "imports_count": 0,
        "has_debug": False,
        "has_tls": False,
        "entry_point": None,
        "entry_point_section": None,
        "suspicious_section_names": [],
        "overlay_size": 0,
        "overlay_ratio": 0.0,
        "executable_writable_sections": [],
        "parse_error": None,
    }

    try:
        pe = pefile.PE(file_path, fast_load=False)
        result["is_pe"] = True
        result["num_sections"] = len(pe.sections)
        result["entry_point"] = int(pe.OPTIONAL_HEADER.AddressOfEntryPoint)

        section_names: list[str] = []
        section_entropies: list[float] = []
        suspicious_names: list[str] = []
        high_entropy_sections: list[str] = []
        exec_write_sections: list[str] = []

        for section in pe.sections:
            name = _safe_section_name(section)
            section_names.append(name)

            data = section.get_data()
            entropy = round(shannon_entropy(data), 4)
            section_entropies.append(entropy)

            lower_name = name.lower()
            if (
                lower_name in KNOWN_SUSPICIOUS_SECTION_NAMES
                or "upx" in lower_name
                or "pack" in lower_name
                or "vmprotect" in lower_name
                or "themida" in lower_name
            ):
                suspicious_names.append(name)

            if entropy >= 7.2:
                high_entropy_sections.append(name)

            flags = _section_flags(section)
            if flags["executable"] and flags["writable"]:
                exec_write_sections.append(name)

        result["section_names"] = section_names
        result["section_entropies"] = section_entropies
        result["avg_section_entropy"] = round(
            sum(section_entropies) / len(section_entropies), 4
        ) if section_entropies else 0.0
        result["max_section_entropy"] = max(section_entropies) if section_entropies else 0.0
        result["high_entropy_sections"] = high_entropy_sections
        result["suspicious_section_names"] = suspicious_names
        result["executable_writable_sections"] = exec_write_sections

        try:
            ep_section = pe.get_section_by_rva(pe.OPTIONAL_HEADER.AddressOfEntryPoint)
            if ep_section is not None:
                result["entry_point_section"] = _safe_section_name(ep_section)
        except Exception:
            result["entry_point_section"] = None

        imports_count = 0
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                if hasattr(entry, "imports") and entry.imports is not None:
                    imports_count += len(entry.imports)
        result["imports_count"] = imports_count

        result["has_debug"] = hasattr(pe, "DIRECTORY_ENTRY_DEBUG")
        result["has_tls"] = hasattr(pe, "DIRECTORY_ENTRY_TLS")

        try:
            overlay_offset = pe.get_overlay_data_start_offset()
            if overlay_offset is not None and overlay_offset < file_size:
                overlay_size = max(0, file_size - int(overlay_offset))
            else:
                overlay_size = 0
        except Exception:
            overlay_size = 0

        result["overlay_size"] = int(overlay_size)
        result["overlay_ratio"] = round((overlay_size / file_size), 4) if file_size > 0 else 0.0

        pe.close()
        return result

    except Exception as exc:
        result["parse_error"] = str(exc)
        return result
