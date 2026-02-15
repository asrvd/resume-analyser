#!/usr/bin/env python3
"""
Professional Resume Generator with Enhanced Formatting
Single-page resumes with proper spacing and professional appearance
"""

import ollama
import pandas as pd
import random
from tqdm import tqdm
import time
from datetime import datetime
import os
from pathlib import Path

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

class ProfessionalResumeGenerator:
    """Generate professional single-page resumes with clean formatting"""
    
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        
        # Create output directories
        Path("output/csv").mkdir(parents=True, exist_ok=True)
        Path("output/pdfs").mkdir(parents=True, exist_ok=True)
        
        # Test model
        try:
            ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': 'Hi'}])
            print(f"✓ Model {model_name} loaded successfully!")
        except Exception as e:
            print(f"✗ Error: {e}")
            raise
        
        # Categories with characteristics
        self.categories = {
            'Private Job': {
                'cgpa_range': (7.0, 9.0),
                'skills_count': (8, 12),
                'internships': (1, 2),
                'projects': (2, 3),
                'certifications': (1, 3),
                'research_papers': (0, 0)
            },
            'Higher Studies': {
                'cgpa_range': (8.0, 9.8),
                'skills_count': (6, 10),
                'internships': (0, 1),
                'projects': (2, 3),
                'certifications': (1, 2),
                'research_papers': (0, 1)
            },
            'Research Field': {
                'cgpa_range': (8.5, 9.9),
                'skills_count': (7, 10),
                'internships': (0, 1),
                'projects': (1, 2),
                'certifications': (0, 2),
                'research_papers': (1, 3)
            },
            'Skill Improvement': {
                'cgpa_range': (5.5, 7.5),
                'skills_count': (3, 6),
                'internships': (0, 1),
                'projects': (1, 2),
                'certifications': (0, 1),
                'research_papers': (0, 0)
            }
        }
        
        # Expanded name database
        self.first_names = [
            # Male names
            'Rahul', 'Amit', 'Vikram', 'Arjun', 'Karan', 'Rohan', 'Aditya', 'Siddharth',
            'Akash', 'Nikhil', 'Varun', 'Raj', 'Vishal', 'Ankit', 'Gaurav', 'Harsh',
            'Kunal', 'Mohit', 'Naman', 'Pranav', 'Rishi', 'Sahil', 'Tanmay', 'Utkarsh',
            'Abhishek', 'Aman', 'Ashish', 'Deepak', 'Hitesh', 'Kartik', 'Manish', 'Naveen',
            'Piyush', 'Sagar', 'Shubham', 'Tarun', 'Yash', 'Aakash', 'Dhruv', 'Ishaan',
            # Female names
            'Priya', 'Sneha', 'Anjali', 'Divya', 'Neha', 'Pooja', 'Shreya', 'Kavya',
            'Riya', 'Meera', 'Isha', 'Ananya', 'Sakshi', 'Diya', 'Ishita', 'Simran',
            'Aditi', 'Nidhi', 'Pallavi', 'Swati', 'Tanvi', 'Vrinda', 'Aishwarya', 'Bhavna',
            'Khushi', 'Mansi', 'Nikita', 'Ritika', 'Sonal', 'Tanya', 'Vidya', 'Zoya',
            'Deepika', 'Gauri', 'Jyoti', 'Komal', 'Megha', 'Payal', 'Sonali', 'Vani'
        ]
        
        self.last_names = [
            'Sharma', 'Kumar', 'Singh', 'Patel', 'Gupta', 'Reddy', 'Iyer', 'Das',
            'Mehta', 'Joshi', 'Nair', 'Verma', 'Malhotra', 'Kapoor', 'Rao', 'Pillai',
            'Agarwal', 'Banerjee', 'Chatterjee', 'Desai', 'Ghosh', 'Khan', 'Menon', 'Mishra',
            'Pandey', 'Roy', 'Saxena', 'Shah', 'Thakur', 'Varma', 'Yadav', 'Bhatt',
            'Chopra', 'Dutta', 'Goel', 'Jain', 'Kulkarni', 'Mukherjee', 'Raman', 'Sinha',
            'Trivedi', 'Agnihotri', 'Bose', 'Dey', 'Ganguly', 'Hegde', 'Krishnan', 'Mittal',
            'Prasad', 'Rane', 'Shetty', 'Tandon', 'Venkatesh', 'Arora', 'Bhatia', 'Dua'
        ]
        
        # Expanded college database
        self.colleges = [
            # West Bengal
            'Kalyani Government Engineering College',
            'Jadavpur University',
            'Heritage Institute of Technology',
            'Techno India University',
            'Haldia Institute of Technology',
            'Narula Institute of Technology',
            'Meghnad Saha Institute of Technology',
            'Institute of Engineering and Management',
            'University of Engineering and Management',
            'Netaji Subhash Engineering College',
            
            # NITs
            'NIT Durgapur',
            'NIT Agartala',
            'NIT Silchar',
            'NIT Rourkela',
            'NIT Patna',
            'NIT Jamshedpur',
            
            # IITs
            'IIT Kharagpur',
            'IIT BHU Varanasi',
            'IIT Guwahati',
            
            # Other states
            'BITS Pilani',
            'VIT Vellore',
            'Manipal Institute of Technology',
            'SRM Institute of Science and Technology',
            'Amity University',
            'Lovely Professional University',
            'Thapar Institute of Engineering',
            'PES University',
            'RV College of Engineering',
            'BMS College of Engineering',
            'Delhi Technological University',
            'NSIT Delhi',
            'IIIT Hyderabad',
            'IIIT Bangalore',
            'Pune Institute of Computer Technology',
            'College of Engineering Pune',
            'Nirma University',
            'DA-IICT Gandhinagar'
        ]
        
        self.branches = [
            'Computer Science and Engineering',
            'Information Technology',
            'Electronics and Communication Engineering',
            'Electrical Engineering',
            'Mechanical Engineering',
            'Civil Engineering',
            'Computer Engineering',
            'Software Engineering',
            'Data Science and Engineering',
            'Artificial Intelligence and Machine Learning'
        ]
        
        # Year variations
        self.year_ranges = [
            '2020-2024',
            '2021-2025',
            '2022-2026',
            '2023-2027',
        ]
        
        # Skills database
        self.tech_skills = {
            'languages': ['Python', 'Java', 'JavaScript', 'C++', 'C', 'Go', 'Rust', 'TypeScript', 
                         'PHP', 'Ruby', 'Kotlin', 'Swift', 'R', 'Scala', 'Dart'],
            'web_frontend': ['React', 'Angular', 'Vue.js', 'HTML5', 'CSS3', 'Bootstrap', 
                           'Tailwind CSS', 'jQuery', 'Redux', 'Next.js', 'Svelte'],
            'web_backend': ['Node.js', 'Django', 'Flask', 'Spring Boot', 'Express.js', 
                          'FastAPI', 'Laravel', 'Ruby on Rails', 'ASP.NET'],
            'mobile': ['React Native', 'Flutter', 'Android', 'iOS', 'Kotlin', 'Swift', 'Xamarin'],
            'databases': ['MySQL', 'MongoDB', 'PostgreSQL', 'Redis', 'SQLite', 'Oracle', 
                        'Cassandra', 'Firebase', 'DynamoDB'],
            'ml_ai': ['Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 
                     'Scikit-learn', 'Keras', 'NLP', 'Computer Vision', 'OpenCV', 'Pandas', 'NumPy'],
            'cloud': ['AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins', 
                    'Terraform', 'Ansible'],
            'tools': ['Git', 'GitHub', 'GitLab', 'VS Code', 'Postman', 'JIRA', 'Linux', 
                    'Figma', 'Selenium']
        }
        
        self.companies = [
            'TCS', 'Infosys', 'Wipro', 'Cognizant', 'Tech Mahindra', 'HCL', 'Accenture',
            'IBM', 'Amazon', 'Microsoft', 'Google', 'Oracle', 'SAP', 'Adobe', 'Cisco',
            'Flipkart', 'Paytm', 'PhonePe', 'Swiggy', 'Zomato', 'Razorpay', 'CRED', 'Meesho',
            'Ola', 'Uber', 'MakeMyTrip', 'OYO', 'Byju\'s', 'Unacademy', 'upGrad',
            'Freshworks', 'Zoho', 'Chargebee', 'Postman', 'BrowserStack',
            'TechVerse Solutions', 'InnovateTech', 'DataCorp Analytics', 'CloudNine Systems',
            'CodeCraft Labs', 'ByteBridge Technologies', 'NexGen Software', 'AlphaWave Tech'
        ]
        
        self.project_types = [
            'E-commerce Platform', 'Social Media Application', 'Task Management System',
            'Real-time Chat Application', 'Food Delivery App', 'Hotel Booking Portal',
            'Online Learning Platform', 'Healthcare Management System', 'Expense Tracker',
            'Weather Forecasting App', 'Movie Recommendation System', 'Blog Platform',
            'Inventory Management', 'Library Management System', 'Attendance System',
            'Stock Price Predictor', 'Sentiment Analysis Tool', 'Image Classification Model',
            'Music Streaming App', 'Fitness Tracking App', 'Recipe Sharing Platform',
            'Job Portal', 'News Aggregator', 'Quiz Application', 'Portfolio Website',
            'Chatbot System', 'URL Shortener', 'To-Do List App', 'Calendar Application'
        ]
        
        self.certifications = [
            'AWS Certified Cloud Practitioner',
            'AWS Certified Solutions Architect',
            'Google Cloud Associate',
            'Microsoft Azure Fundamentals',
            'Oracle Certified Java Programmer',
            'Python for Data Science (Coursera)',
            'Machine Learning Specialization (Coursera)',
            'Deep Learning Specialization (Coursera)',
            'Full Stack Web Development (Udemy)',
            'React - The Complete Guide (Udemy)',
            'Java Programming Masterclass (Udemy)',
            'Docker and Kubernetes (Udemy)',
            'MongoDB Developer Certification',
            'Cisco CCNA',
            'CompTIA Security+',
            'NPTEL - Data Structures and Algorithms',
            'NPTEL - Database Management Systems',
            'NPTEL - Cloud Computing',
            'Google Data Analytics Certificate',
            'IBM Data Science Professional Certificate',
            'Meta Front-End Developer Certificate',
            'HackerRank Problem Solving Certificate',
            'HackerRank Python Certificate',
            'Red Hat Certified System Administrator'
        ]
    
    def generate_resume_data(self, category, resume_count=1):
        """Generate structured resume data"""
        
        specs = self.categories[category]
        
        name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
        email = f"{name.lower().replace(' ', '.')}@gmail.com"
        phone = f"+91-{random.randint(7000000000, 9999999999)}"
        
        cgpa = round(random.uniform(*specs['cgpa_range']), 2)
        college = random.choice(self.colleges)
        branch = random.choice(self.branches)
        year = random.choice(self.year_ranges)
        
        # LinkedIn (50% have it)
        linkedin = None
        if random.random() > 0.5:
            linkedin = f"linkedin.com/in/{name.lower().replace(' ', '-')}"
        
        # GitHub (60% have it)
        github = None
        if random.random() > 0.4:
            github = f"github.com/{name.lower().replace(' ', '')}"
        
        # Generate skills
        num_skills = random.randint(*specs['skills_count'])
        skills = self.select_skills(num_skills, category)
        
        # Generate projects
        projects = []
        num_projects = random.randint(*specs['projects'])
        
        for i in range(num_projects):
            project_skills = random.sample(skills, min(3, len(skills)))
            projects.append({
                'name': random.choice(self.project_types),
                'duration': f"{random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])} {random.randint(2023, 2024)} - {random.choice(['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])} {random.randint(2023, 2024)}",
                'description': [
                    f"Developed using {', '.join(project_skills[:2])}",
                    f"Implemented {random.choice(['authentication', 'real-time features', 'REST APIs', 'payment integration', 'user dashboard'])}",
                ]
            })
        
        # Generate internships
        internships = []
        num_internships = random.randint(*specs['internships'])
        
        for i in range(num_internships):
            intern_skills = random.sample(skills, min(2, len(skills)))
            internships.append({
                'company': random.choice(self.companies),
                'role': random.choice(['Software Development Intern', 'Web Developer Intern', 
                                      'Data Analyst Intern', 'Backend Developer Intern',
                                      'Frontend Developer Intern', 'ML Engineer Intern']),
                'duration': f"{random.choice(['May', 'Jun', 'Jul'])} {random.randint(2023, 2024)} - {random.choice(['Aug', 'Sep', 'Oct'])} {random.randint(2023, 2024)}",
                'description': [
                    f"Worked on {random.choice(['backend APIs', 'frontend features', 'data analysis', 'testing automation', 'database optimization'])} using {intern_skills[0]}",
                    f"Collaborated with {random.choice(['development', 'product', 'design', 'QA'])} team"
                ]
            })
        
        # Generate certifications
        certifications = []
        num_certs = random.randint(*specs['certifications'])
        certifications = random.sample(self.certifications, min(num_certs, len(self.certifications)))
        
        # Generate research papers
        papers = []
        num_papers = random.randint(*specs['research_papers'])
        
        for i in range(num_papers):
            papers.append({
                'title': f"Study on {random.choice(['Machine Learning', 'IoT Systems', 'Blockchain', 'AI', 'Deep Learning', 'Cloud Computing', 'Cybersecurity'])} Applications",
                'conference': random.choice(['IEEE', 'Springer', 'ACM', 'ScienceDirect']),
                'year': random.randint(2023, 2024)
            })
        
        return {
            'category': category,
            'name': name,
            'email': email,
            'phone': phone,
            'linkedin': linkedin,
            'github': github,
            'cgpa': cgpa,
            'college': college,
            'branch': branch,
            'year': year,
            'skills': skills,
            'projects': projects,
            'internships': internships,
            'certifications': certifications,
            'research_papers': papers
        }
    
    def select_skills(self, num_skills, category):
        """Select appropriate skills"""
        all_skills = []
        
        # Languages (always 2-3)
        all_skills.extend(random.sample(self.tech_skills['languages'], min(3, num_skills)))
        remaining = num_skills - len(all_skills)
        
        if category == 'Private Job':
            all_skills.extend(random.sample(self.tech_skills['web_frontend'], min(2, remaining)))
            all_skills.extend(random.sample(self.tech_skills['web_backend'], min(2, remaining)))
            all_skills.extend(random.sample(self.tech_skills['databases'], min(1, remaining)))
        elif category == 'Research Field':
            all_skills.extend(random.sample(self.tech_skills['ml_ai'], min(3, remaining)))
        elif category == 'Higher Studies':
            all_skills.extend(random.sample(self.tech_skills['ml_ai'], min(2, remaining)))
            all_skills.extend(random.sample(self.tech_skills['web_frontend'], min(1, remaining)))
        
        remaining = num_skills - len(all_skills)
        if remaining > 0:
            all_skills.extend(random.sample(self.tech_skills['tools'], min(remaining, len(self.tech_skills['tools']))))
        
        return list(set(all_skills))[:num_skills]
    
    def create_pdf_resume(self, data, output_path):
        """Create professional single-page PDF resume with better spacing"""
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.6*inch,
            bottomMargin=0.5*inch
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles - Professional fonts and spacing
        name_style = ParagraphStyle(
            'NameStyle',
            parent=styles['Normal'],
            fontSize=20,
            textColor=colors.black,
            spaceAfter=6,
            spaceBefore=0,
            alignment=TA_CENTER,
            fontName='Times-Bold',
            leading=24
        )
        
        contact_style = ParagraphStyle(
            'ContactStyle',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Times-Roman',
            leading=11
        )
        
        section_heading_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=6,
            spaceBefore=10,
            fontName='Times-Bold',
            alignment=TA_LEFT,
            leading=13
        )
        
        item_title_style = ParagraphStyle(
            'ItemTitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=3,
            fontName='Times-Bold',
            leading=12
        )
        
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=4,
            leading=12,
            fontName='Times-Roman'
        )
        
        bullet_style = ParagraphStyle(
            'BulletStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=3,
            leading=12,
            leftIndent=15,
            fontName='Times-Roman'
        )
        
        # === HEADER (Centered) ===
        story.append(Paragraph(data['name'].upper(), name_style))
        
        # Contact info
        contact_parts = [data['phone'], data['email']]
        if data['linkedin']:
            contact_parts.append(data['linkedin'])
        if data['github']:
            contact_parts.append(data['github'])
        
        contact_text = " | ".join(contact_parts)
        story.append(Paragraph(contact_text, contact_style))
        
        # Horizontal line separator
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=10))
        
        # === EDUCATION ===
        story.append(Paragraph("EDUCATION", section_heading_style))
        
        edu_lines = [
            f"<b>{data['college']}</b>",
            f"Bachelor of Technology in {data['branch']}",
            f"{data['year']} | CGPA: <b>{data['cgpa']}/10.0</b>"
        ]
        
        for line in edu_lines:
            story.append(Paragraph(line, body_style))
        
        story.append(Spacer(1, 0.1*inch))
        
        # === TECHNICAL SKILLS ===
        if data['skills']:
            story.append(Paragraph("TECHNICAL SKILLS", section_heading_style))
            skills_text = ", ".join(data['skills'])
            story.append(Paragraph(skills_text, body_style))
            story.append(Spacer(1, 0.1*inch))
        
        # === PROJECTS ===
        if data['projects']:
            story.append(Paragraph("PROJECTS", section_heading_style))
            
            for idx, project in enumerate(data['projects']):
                project_title = f"<b>{project['name']}</b> | <i>{project['duration']}</i>"
                story.append(Paragraph(project_title, item_title_style))
                
                for desc in project['description']:
                    story.append(Paragraph(f"• {desc}", bullet_style))
                
                # Add small space between projects
                if idx < len(data['projects']) - 1:
                    story.append(Spacer(1, 0.05*inch))
            
            story.append(Spacer(1, 0.1*inch))
        
        # === EXPERIENCE ===
        if data['internships']:
            story.append(Paragraph("EXPERIENCE", section_heading_style))
            
            for idx, internship in enumerate(data['internships']):
                intern_title = f"<b>{internship['role']}</b> | {internship['company']}"
                story.append(Paragraph(intern_title, item_title_style))
                
                duration_text = f"<i>{internship['duration']}</i>"
                story.append(Paragraph(duration_text, body_style))
                
                for desc in internship['description']:
                    story.append(Paragraph(f"• {desc}", bullet_style))
                
                # Add small space between internships
                if idx < len(data['internships']) - 1:
                    story.append(Spacer(1, 0.05*inch))
            
            story.append(Spacer(1, 0.1*inch))
        
        # === CERTIFICATIONS ===
        if data['certifications']:
            story.append(Paragraph("CERTIFICATIONS", section_heading_style))
            
            for cert in data['certifications']:
                story.append(Paragraph(f"• {cert}", bullet_style))
            
            story.append(Spacer(1, 0.1*inch))
        
        # === RESEARCH PUBLICATIONS ===
        if data['research_papers']:
            story.append(Paragraph("RESEARCH PUBLICATIONS", section_heading_style))
            
            for paper in data['research_papers']:
                paper_text = f"• <b>{paper['title']}</b>, {paper['conference']} {paper['year']}"
                story.append(Paragraph(paper_text, bullet_style))
        
        # Build PDF
        try:
            doc.build(story)
        except Exception as e:
            print(f"Error creating PDF: {e}")
    
    def resume_to_text(self, data):
        """Convert to plain text"""
        text_parts = []
        
        text_parts.append(data['name'].upper())
        text_parts.append(f"{data['email']} | {data['phone']}")
        if data['linkedin']:
            text_parts.append(f"LinkedIn: {data['linkedin']}")
        if data['github']:
            text_parts.append(f"GitHub: {data['github']}")
        text_parts.append("")
        
        text_parts.append("EDUCATION")
        text_parts.append(data['college'])
        text_parts.append(f"Bachelor of Technology in {data['branch']}")
        text_parts.append(f"{data['year']} | CGPA: {data['cgpa']}/10.0")
        text_parts.append("")
        
        if data['skills']:
            text_parts.append("TECHNICAL SKILLS")
            text_parts.append(", ".join(data['skills']))
            text_parts.append("")
        
        if data['projects']:
            text_parts.append("PROJECTS")
            for project in data['projects']:
                text_parts.append(f"{project['name']} ({project['duration']})")
                for desc in project['description']:
                    text_parts.append(f"- {desc}")
            text_parts.append("")
        
        if data['internships']:
            text_parts.append("EXPERIENCE")
            for internship in data['internships']:
                text_parts.append(f"{internship['role']} | {internship['company']}")
                text_parts.append(internship['duration'])
                for desc in internship['description']:
                    text_parts.append(f"- {desc}")
            text_parts.append("")
        
        if data['certifications']:
            text_parts.append("CERTIFICATIONS")
            for cert in data['certifications']:
                text_parts.append(f"- {cert}")
            text_parts.append("")
        
        if data['research_papers']:
            text_parts.append("RESEARCH PUBLICATIONS")
            for paper in data['research_papers']:
                text_parts.append(f"- {paper['title']}, {paper['conference']} {paper['year']}")
            text_parts.append("")
        
        return "\n".join(text_parts)
    
    def generate_dataset(self, total_resumes=500, distribution=None, generate_pdfs=True):
        """Generate complete dataset"""
        
        print(f"\n{'='*70}")
        print(f"PROFESSIONAL RESUME GENERATION")
        print(f"{'='*70}")
        print(f"Total Resumes: {total_resumes}")
        print(f"Generate PDFs: {generate_pdfs}")
        print(f"{'='*70}\n")
        
        if distribution is None:
            distribution = {
                'Private Job': total_resumes // 4,
                'Higher Studies': total_resumes // 4,
                'Research Field': total_resumes // 4,
                'Skill Improvement': total_resumes // 4
            }
        
        print("Distribution:")
        for cat, count in distribution.items():
            print(f"  {cat}: {count}")
        print()
        
        all_resumes = []
        category_counters = {cat: 1 for cat in distribution.keys()}
        start_time = time.time()
        
        for category, count in distribution.items():
            print(f"\nGenerating: {category} ({count} resumes)")
            
            for i in tqdm(range(count), desc=category):
                data = self.generate_resume_data(category, resume_count=category_counters[category])
                text = self.resume_to_text(data)
                
                # PDF filename: firstname_lastname_count.pdf
                pdf_path = None
                if generate_pdfs:
                    first_name = data['name'].split()[0].lower()
                    last_name = data['name'].split()[-1].lower()
                    pdf_filename = f"{first_name}_{last_name}_{category_counters[category]}.pdf"
                    pdf_path = f"output/pdfs/{pdf_filename}"
                    self.create_pdf_resume(data, pdf_path)
                
                all_resumes.append({
                    'id': f"RESUME_{len(all_resumes)+1:04d}",
                    'text': text,
                    'category': category,
                    'name': data['name'],
                    'email': data['email'],
                    'phone': data['phone'],
                    'cgpa': data['cgpa'],
                    'college': data['college'],
                    'branch': data['branch'],
                    'year': data['year'],
                    'num_skills': len(data['skills']),
                    'num_projects': len(data['projects']),
                    'num_internships': len(data['internships']),
                    'num_certifications': len(data['certifications']),
                    'num_research_papers': len(data['research_papers']),
                    'pdf_path': pdf_path,
                    'generated_at': datetime.now().isoformat()
                })
                
                category_counters[category] += 1
                time.sleep(0.05)
        
        elapsed = time.time() - start_time
        
        df = pd.DataFrame(all_resumes)
        csv_path = f'output/csv/synthetic_resumes_{total_resumes}.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Total: {len(df)} resumes")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"CSV: {csv_path}")
        if generate_pdfs:
            print(f"PDFs: output/pdfs/ ({len(df)} files)")
        print(f"{'='*70}\n")
        
        print("Statistics:")
        print(df['category'].value_counts())
        print(f"\nCGPA by Category:")
        print(df.groupby('category')['cgpa'].mean().round(2))
        
        return df


def main():
    """Interactive generation"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║        PROFESSIONAL SINGLE-PAGE RESUME GENERATOR                 ║
    ║           Enhanced Spacing & Professional Fonts                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    total = int(input("Number of resumes (default 100): ").strip() or "100")
    pdfs = input("Generate PDFs? (y/n, default y): ").strip().lower() != 'n'
    
    print(f"\nGenerating {total} resumes...")
    
    gen = ProfessionalResumeGenerator(model_name="llama3.2:3b")
    df = gen.generate_dataset(total_resumes=total, generate_pdfs=pdfs)
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()