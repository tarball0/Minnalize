# Minnalize

Minnalize is a binary visualization and malware classification tool developed for the ACM MITS Hackathon. It performs malware triage by combining static PE header analysis, Authenticode signature verification, and computer vision using a fine-tuned Convolutional Neural Network (CNN).

The project converts executable binaries into grayscale images to identify structural patterns and anomalies that traditional static analysis might miss.

## Features

*   Binary-to-Image Conversion: Transforms .exe files into grayscale images using Nataraj-style width mapping. This process preserves the spatial correlation of byte sequences, making malicious patterns visible to vision models.
*   Deep Learning Classification: Employs a fine-tuned EfficientNet-B0 model (PyTorch) trained on image representations of both benign and malicious binary samples.
*   Static PE Analysis: Extracts features such as section entropy, import counts, and suspicious API calls (e.g., VirtualAlloc, LoadLibrary) using the pefile library.
*   Authenticode Signature Verification: Utilizes Windows PowerShell features (Get-AuthenticodeSignature) to verify file integrity and check for trusted publishers.
*   Hybrid Scoring Engine: Aggregates visual signals, static rules, and signature metadata into a final suspicion score (0-100).
*   Report Generation: Provides human-readable explanations for the assigned risk level, detailing specific PE anomalies or CNN confidence margins.
*   Electron Interface: A desktop UI built with Electron for file ingestion and result visualization.

## Technical Stack

*   Frontend: Electron, JavaScript, HTML, CSS.
*   Backend: Python 3.
*   Libraries: PyTorch, Torchvision, NumPy, Pillow, pefile.
*   System Integration: Windows PowerShell.

## Architecture

1.  Ingestion: The user selects a file via the Electron UI.
2.  Signature Check: The system first checks for an Authenticode signature. Valid signatures from trusted publishers (e.g., Microsoft, Google) significantly reduce the suspicion score.
3.  PE Parsing: The PE header is parsed for structural anomalies, high entropy (indicating packing or encryption), and suspicious imports.
4.  Visualization: The binary is mapped to a grayscale image. The image width is adjusted based on file size to maintain consistent pattern density.
5.  CNN Inference: If the file is unsigned or suspicious, the image is passed to EfficientNet-B0 to detect malicious visual fingerprints.
6.  Fusion: The final verdict uses a weighted blend of the CNN output (70%) and PE rules (30%) for unsigned files.

## Authors

*   Chris Paul (tarball0)
*   Fahad (Fahad-uz)
*   Aidan Jason (AidanJ07)

## Requirements

*   OS: Windows (Required for full PowerShell signature verification).
*   Python 3.8+ with torch, torchvision, pefile, and numpy.
*   Node.js for the Electron frontend.

Developed for the ACM MITS Hackathon.
