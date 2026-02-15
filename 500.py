from generate_pdf_resumes import ProfessionalResumeGenerator

# Create generator
gen = ProfessionalResumeGenerator(model_name="llama3.2:3b")

# Generate 500 resumes (text only - faster)
df = gen.generate_dataset(
    total_resumes=500, generate_pdfs=True  # Set to True if you want PDFs
)

print("\n✓ Done! Check output/csv/ for the dataset")
