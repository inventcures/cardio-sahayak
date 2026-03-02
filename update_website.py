import os

projects_file = os.path.expanduser("~/showmethecode/inventcures.github.io/_pages/projects.md")
with open(projects_file, "r", encoding="utf-8") as f:
    content = f.read()

card_html = """
<!-- Cardio-Sahayak India - NEW -->
<section class="featured-section">
  <a href="/cardio-sahayak/" style="display: block; background: linear-gradient(135deg, #1f0510 0%, #4c0519 100%); border: 2px solid #be123c; border-radius: 12px; overflow: hidden; text-decoration: none; position: relative; margin-bottom: 2rem;">
    <div style="position: absolute; top: -8px; right: 20px; background: linear-gradient(135deg, #f43f5e, #be123c); color: #fff; font-size: 0.65rem; font-weight: 700; padding: 0.3rem 0.6rem; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.05em; box-shadow: 0 4px 12px rgba(190, 18, 60, 0.4);">New</div>
    <div style="padding: 1.5rem 2rem;">
      <div style="display: inline-block; background: #be123c; color: #fff; font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;">AI + Cardiology</div>
      <h3 style="color: #fff; font-size: 1.5rem; margin: 0 0 0.4rem 0; font-weight: 700;">Cardio-Sahayak India 🇮🇳 🫀</h3>
      <p style="color: #fda4af; font-size: 0.9rem; margin: 0 0 0.75rem 0; font-weight: 500;">Multimodal Foundation Model for South Asian Cardiology</p>
      <p style="color: #ddd; font-size: 0.85rem; line-height: 1.5; margin: 0 0 1rem 0;">A dual-architecture LLM and Vision-Language Model (MedGemma-27B + MedSigLIP) fine-tuned for complex cardiology care. Capable of deep clinical reasoning and native 12-lead ECG interpretation, optimized specifically for the unique South Asian phenotypic markers like lower BMI thresholds and the MYBPC3 Δ25bp variant.</p>
      <div style="display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem;">
        <span style="background: rgba(244,63,94,0.15); color: #fb7185; padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">LLM + VLM</span>
        <span style="background: rgba(244,63,94,0.15); color: #fb7185; padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">MedGemma-27B</span>
        <span style="background: rgba(244,63,94,0.15); color: #fb7185; padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">MedSigLIP</span>
        <span style="background: rgba(244,63,94,0.15); color: #fb7185; padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">ECG Interpretation</span>
      </div>
      <span style="color: #fda4af; font-weight: 600; font-size: 0.9rem;">View Project & Preprint →</span>
    </div>
  </a>
</section>
"""

if "Cardio-Sahayak" not in content:
    # Insert right before Virtual Tumor Board
    insertion_point = "<!-- Virtual Tumor Board - Featured -->"
    content = content.replace(insertion_point, card_html + "\n" + insertion_point)
    with open(projects_file, "w", encoding="utf-8") as f:
        f.write(content)
    print("Added Cardio-Sahayak card to projects.md")

cardio_file = os.path.expanduser("~/showmethecode/inventcures.github.io/_pages/cardio-sahayak.md")
with open(cardio_file, "r", encoding="utf-8") as f:
    content_c = f.read()

new_content_c = content_c.replace("v1_cardio-sahayak_preprint.pdf", "v2_cardio-sahayak_preprint.pdf")

if content_c != new_content_c:
    with open(cardio_file, "w", encoding="utf-8") as f:
        f.write(new_content_c)
    print("Updated preprint link to v2 in cardio-sahayak.md")
