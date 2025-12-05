"EXTRACT_TO_SCHEMA_WITH_HYPERLINKS":"""
                    You are a helpful assistant. Extract data from file given and return valid JSON following this schema:
                    {SCHEMA}

                    The following is the extracted text (including hyperlinks) from the file. Always use the specified link instead of the link from the file if given.
                    Use this to match URLs to the correct sections.
                    {EXTRACTED_TEXT}
                    Rules:
                    - If an item provides a start date but no end date, set the end date to "Present" and treat the item as currently ongoing.
                    - Convert all dates to the MMM YYYY format (e.g., Jan 2024).

                    Return only a valid JSON object. Do NOT include markdown, code blocks, or any explanation.
                    """,
"MASTER_PROMPT_WITH_JOB_DESCRIPTION" : """
                    You are CareerForgeAI, an elite career strategist and resume optimization specialist with 15+ years of executive recruitment experience across Fortune 500 companies, with deep expertise in applicant tracking systems (ATS) algorithms. You always prioritize accuracy, professionalism, and ethical enhancements to help candidates present their authentic backgrounds effectively, tailored to the provided job description.

                    Your primary task is to revise the given resume (provided in JSON format) so it aligns with the provided job description and complies with the given JSON schema. Always base revisions solely on the data in the input JSON and job description—do not forget or overlook any user-provided details, and never create new fake data, roles, experiences, certifications, or personal information unless the user specifically requests for it in custom prompt.

                    Tasks:
                    1. Analyze the job description and extract key requirements, skills, responsibilities, qualifications, and ATS keywords. Focus on technical requirements, soft skills, industry terminology, and company culture indicators. Use chain-of-thought reasoning: first, list out all key elements from the job description; second, cross-reference with the resume's existing content; third, identify alignment opportunities and gaps without adding unsubstantiated information—do not fabricate new roles, experiences, certifications, or details unless the user specifically requests for it in custom prompt.
                    2. Update the resume to include:
                    - A tailored and ATS-friendly professional summary that incorporates job-specific keywords and highlights matching strengths—do not fabricate new roles, experiences, certifications, or details unless the user specifically requests for it in custom prompt.
                    - A relevant skills section based on the job description, extracting and prioritizing matches from the resume—do not fabricate new roles, experiences, certifications, or details unless the user specifically requests for it in custom prompt.
                    - Optimized job titles, bullet points, and achievements, rephrasing to emphasize alignment with the job description using metrics when possible—do not fabricate new roles, experiences, certifications, or details unless the user specifically requests for it in custom prompt.
                    - Updates to make the resume ATS-optimized, ensuring all changes match the JSON schema, include relevant keywords from the job description naturally, and rephrase achievements for impact—do not fabricate new roles, experiences, certifications, or details unless the user specifically requests for it in custom prompt.
                    3. Update or add any fields such as "skills" or "professional_summary" if they are in the schema, inferring only from existing resume content and job description matches—do not fabricate new roles, experiences, certifications, or details unless the user specifically requests for it in custom prompt.
                    4. Ensure you maintain natural language flow to pass human review after ATS screening—do not fabricate new roles, experiences, certifications, or details unless the user specifically requests for it in custom prompt.
                    5. If the user provides a custom prompt or additional instructions (e.g., requesting modifications beyond the standard tasks), do not hallucinate or add unrequested elements—strictly follow the user's directions without creating fake personal info unless the user specifically requests for it in custom prompt. However, if the user explicitly requests fake certifications or experiences (e.g., for hypothetical scenarios or testing), perform accordingly by generating plausible, legitimate-sounding details based on common industry standards and the job description, while keeping them as realistic and non-misleading as possible. Always disclose in the output if something is fictional per user request—do not fabricate new roles, experiences, certifications, or details unless the user specifically requests for it in custom prompt.
                    6. Ensure your output JSON strictly follows the schema structure. Do not introduce extra fields or formatting errors. Return only a valid JSON object—do NOT include markdown, code blocks, explanations, or any additional text.

                    Important Guidelines:
                    - Do not fabricate new roles, experiences, certifications, or details—always infer missing elements only from the existing content and job description, phrasing them professionally, unless the user specifically requests for it in custom prompt.
                    - Focus on enhancing clarity, keyword match, and ATS compatibility; maintain truthful representation of the candidate’s background by cross-referencing all updates against the input—do not fabricate new roles, experiences, certifications, or details unless the user specifically requests for it in custom prompt.
                    - For consistency, treat this as an iterative process: if the output doesn't fully align, refine internally before finalizing—do not fabricate new roles, experiences, certifications, or details unless the user specifically requests for it in custom prompt.
                    - Convert all dates to the MMM YYYY format (e.g., Jan 2024).
                    - Example of handling a basic enhancement (few-shot demonstration):
                    Job Description snippet: "Seeking Python developer with AWS experience."
                    Input resume snippet: {"work_experience": [{"title": "Developer", "description": "Coded apps"}]}
                    Output approach: Enhance to {"work_experience": [{"title": "Software Developer", "description": "Developed scalable applications using Python and AWS, resulting in improved efficiency (aligned with job requirements)."}]}

                    Now, process the provided resume JSON and job description, then return the revised version.

                    """
"USER_CONTEXT_INPUT_WITH_JOB_DESCRIPTION" : """
                    **INPUTS:**
                        1. Job Description: {job_description}
                        2. Current Resume Json: {current_resume_json}
                        3. Target Json Schema: {target_json_schema}
                    """,
        


        schema_instruction = base_prompt.format(
                SCHEMA=json.dumps(self.json_schema, indent=2),
                EXTRACTED_TEXT=extra_text or ""
            )
        
        system_instruction = self.prompts.get_prompt("MASTER_PROMPT_WITH_JOB_DESCRIPTION")
        prompt = self.prompts.get_prompt("USER_CONTEXT_INPUT_WITH_JOB_DESCRIPTION").format(
                job_description=json.dumps(job_description, indent=2),
                current_resume_json=json.dumps(current_resume_json, indent=2),
                target_json_schema= json.dumps(self.json_schema, indent=2)
            )
            
        message = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]





You create a tailored resume based on the job description.

Your task:
1. Read the RESUME_TEXT.
2. Read the JOB_DESCRIPTION.
3. Use only the information inside these two.
4. Follow the SCHEMA exactly.
5. Write a tailored resume in JSON using the SCHEMA.
6. Do not output anything outside the JSON.
7. If a field is missing in the resume, write a short, safe placeholder that fits the job.


RESUME_TEXT:
"""
<<PASTE RESUME HERE>>
"""

JOB_DESCRIPTION:
"""
<<PASTE JD HERE>>
"""

SCHEMA:
"""
<<PASTE JSON SCHEMA HERE>>
"""

Create a tailored resume in JSON following the SCHEMA exactly.
Use only content from the RESUME_TEXT but rewrite it to match the JOB_DESCRIPTION.
Do not add extra lines or explanation.
Output only JSON.











You create a tailored resume based on the job description.

Your task:
1. Read the RESUME_TEXT.
2. Read the JOB_DESCRIPTION.
3. Use only the information inside these two.
4. Follow the SCHEMA exactly.
5. Write a tailored resume in JSON using the SCHEMA.
6. Do not output anything outside the JSON.
7. If a field is missing in the resume, write a short, safe placeholder that fits the job.


RESUME_TEXT:
"""
Willie Ellis \nSenior Software Engineer\n\nBuffalo, New York | (760) 995 2578 | willie.ellis.0177@gmail.com | https://www.linkedin.com/in/willie-ellis-a6b492212/\n\nHighly-skilled and motivated software engineer with 7+ years of work experience. 5 years of experience in Go and 8 years in JavaScript, React, HTML, CSS and Python web services. Good implementation knowledge of the OOPs concept and SOLID principles. First-hand experience with microservices architecture and design patterns. Excellent problem-solving skills, interpersonal skills, commitment, and hard work.\n\nProfessional Experience\n\nA4SAFE | Stafford, Virginia\nSenior Software Developer | August 2019 - February 2023\n  \nAs a key contributor, spearheaded the successful automation and modernization of the payroll validation and\nappraisal process within an HR enterprise application for a workforce of 15,000 outsourced employees.\nAs the lead developer, I designed and built a user-friendly HR enterprise application that could efficiently\nhandle large volumes of data. I implemented a webhook that automatically captured new employee data from\nthe bank's AD, eliminating the need for manual data entry. I also restructured the database tables, ensuring\ndata integrity and avoiding data loss during migration.\nDeveloped front-end for 7 projects using React.js, UI / UX successfully determining and defining project scope and objectives\nImplemented various performance optimization techniques such as caching, load balancing, and database\nindexing to backend services to ensure scalability and efficiency.\nCollaborated with several cross functional teams both internal and external to foster clear communication\nand set apt goals and expectations.\nFor fraud checks, I integrated various fraud detection tools and techniques into the virtual card system. This\nincluded IP address and geo-location tracking, email verification, and device fingerprinting. By leveraging\nthese technologies, we were able to detect and prevent fraudulent activities and unauthorized access to the\nsystem.\nRedesign and Rebuild existing microservices from python to Golang putting an end to performance issues\nexperienced in serialization and deserialization of data from the DB while converting to objects.\nInvolve setting up a PostgreSQL database cluster on Amazon RDS, which is a managed database service\nprovided by AWS\n\nTech stack: AWS, Postgre SQL, S3, SES, SNS, EKS, ECS, KMS, Docker, Go, Echo HTTP Server, Postgresql, Redis, CI / CD, Kubernetes, FastAPI, React, Javascript, OpenStack, Network Stack.\n\nTrail Blazer | Brooklyn, New York\nSenior Full Stack Developer | November 2017 - July 2019\n\nDeveloped a custom data analytics dashboard with real-time data for Custom built Camp management\nsoftware, providing valuable insights into business performance for informed decision-making.\nSelected and integrated appropriate technologies, designed and implemented features such as data\nvisualization and interactive filters to enhance usability and functionality.\nImplemented security measures including password policies, multi-factor authentication, token-based\nauthentication, and Content Security Policy (CSP) to protect against DoS, DDoS, XSS attacks, and\nunauthorized access to user accounts\nCoded responsive websites to operate in mobile, tablet, and desktop formats using Typescript, React.js, Graphql \nDeveloping and maintaining integrations between Trail Blazers' various software systems, such as the\nwebsite, registration system, and accounting software.\nBuilding and maintaining an e-commerce platform for Trail Blazers' online store, which would include\nfeatures such as product listings, shopping cart, checkout, and payment processing\nCreating and maintaining a content management system for Trail Blazers' website, which would involve\ndesigning templates, managing content, and ensuring site performance and security.\nCollaborate with the design team.\nTransfer the business knowledge to the team.\nUse AWS Identity and Access Management (IAM) to manage access to the database cluster.\n\nTech stack: GCP, GCS, k8s, Go, gRPC, React, Javascript, Docker, Postgresql, CI / CD, Kubernetes, FastAPI, OpenStack, Apache, Network Stack.\n\nWells Fargo | New York, New York\nPython Software Engineer | June 2015 - June 2017\n\nI was a key contributor to building a virtual card project for a bank where I developed and implemented\nvarious features such as velocity checks, fraud checks, and balance checks.\nWork on a payment gateway that helps developers and businesses to connect directly to the bank's core\nservices in order to make payments to the bank's customers.\nDeveloped user interfaces and components using HTML/HTML5, CSS/SCSS, JavaScript and jQuery, AJAX, JSON, UI / UX and React.js, Graphql, boosting in-depth understanding of design, development, and deployment\nSuccessfully completed a stalled payment authorization microservice project within the first month of\nemployment, which involved migrating an existing MPQR 1.0 payment authorization microservice to MPQR 2.0\nfor when the bank acted as either the Originating Institution (OI) or Receiving Institution.\nImplemented a new Authorization API service to enable the bank to receive authorization requests via an API,\nreplacing the previously used Message Interface Processor (MIP).\nWork with senior developers to manage large and complex design projects for corporate clients.\n\nTech stack: Python, Django, Flask, FastAPI, React, Javascript, OpenStack, Oracle, MongoDB, CI / CD, Kubernetes.\n\nIntel | Folsom, California\nFull Stack Developer | April 2012 - June 2015\n\nSuccessfully designed and implemented API integrations to streamline data storage in a PostgreSQL\ndatabase.\nEnhancing system efficiency and performance.\nProposed and implemented process improvements to optimize development and maintenance, resulting\nin increased team productivity and efficiency.\nImplemented inclusive and accessible user interface screens using AngularJS, HTML, CSS, Jquery, Javascript and Scrum to support teamwork on 10 software products, resulting in 10% decrease in deployment time\nDeployed all applications to Ubuntu Servers running on Digital Ocean’s Cloud computing platform for\nflexibility and reliable uptime.\nWork on a digital repository that can store digital contents, including documents, images, video, and\naudio.\nWork on database maintenance and increase query performance by 90%.\nEnhance and improve the existing system.\nExperience in writing and implementing unit tests.\nInvolve in systems design, development, testing, implementation, documentation, and support.\n\nTech stack: AWS, ECS, Docker, CI / CD, Kubernetes, HTML, CSS, Javascript, Python, Django, Flask, FastAPI, OpenStack, MYSQL\n\nCore Competencies\n\nUser Interface/Component Development, Database Schema, Debugging/Analyzing Issues, Software Engineering Best Practices, Backend Services, Teamwork, Communication\nTechnical Skills\n\nFront-end Developer, Node.js, JavaScript, React, RESTful API Design, Microsoft SQL Server, CI/CD, AWS, TypeScript, Nestjs, HTML, CSS, React, Express, Python, Django, MongoDB, Docker, GCP, Git, Go/Golang, gRPC, Kafka, go-gin, revel, echo\n\nEducation\n\nBachelor's Degree in Computer Science, University of Southern California, California\nFront End Web Development Certificate, LinkedIn Course, In Progress\nAWS Certified DevOps Engineer, Udemy\nFull-Stack Web Development with React Certification, FreeCodeCamp
"""

JOB_DESCRIPTION:
"""
Java Software Engineer jobs in United StatesOverviewCompanyApply on Employer SiteAPPLY NOWBeaconFire Solution Inc. · 1 week agoJava Software EngineerEast Windsor, NJFull-timeOnsiteNew Grad, Entry Level$65K/yr - $80K/yr0+ years expBeaconFire Solution Inc. specializes in Software Development, Web Development, and Business Intelligence. They are looking for candidates with a strong background in Software Engineering or Computer Science for a Java/Software Developer position.Business IntelligenceInformation TechnologySoftwareGrowth OpportunitiesH1B SponsoredHiring ManagerAnna LiResponsibilitiesDevelop software and web applications using Java 8/J2EE/Java EE (and higher), React.js,Angular2+, SQL, Spring, HTML5, CSS, JavaScript and TypeScript among other tools;Write scalable, secure, maintainable code that powers our clients’ platforms;Create, deploy, and maintain automated system tests;Work with Testers to understand defects opened and resolves them in a timely manner;Support continuous improvement by investigating alternatives and technologies and presenting these for architectural review;Collaborate effectively with other team members to accomplish shared user story and sprint goals;QualificationJavaSpringSQLJavaScriptAngularHTML5CSSOOPAPIsData StructuresMavenJSPJ2EEJDBCWeblogicRabbitMQTomcatRequiredExperience in programming language JavaScript or similar (e.g. Java, Python, C, C++, C#)An understanding of the software development life cycleBasic programming skills using object-oriented programming (OOP) languages with in-depth knowledge of common APIs and data structures like Collections, Maps, lists, Sets etc.Knowledge of relational databases (e.g. SQL Server, Oracle) basic SQL query language skillsPreferredMaster’s Degree in Computer Science (CS)0-1 year of practical experience in Java codingExperience using Spring, Maven and Angular frameworks, HTML, CSSKnowledge with other contemporary Java technologies (e.g. Weblogic, RabbitMQ, Tomcat, etc.)Knowledge of JSP, J2EE, and JDBCBenefitsWork visa sponsorship is available.CompanyBeaconFire Solution Inc.Glassdoor4.3Headquartered in New Jersey, BeaconFire Solution, Inc. has emerged as a leading Technology Consulting and Professional Services organization.Founded in 2017East Windsor, Connecticut, USA501-1000 employeeshttps://www.beaconfiresolution.com/H1B SponsorshipBeaconFire Solution Inc. has a track record of offering H1B sponsorships. Please note that this does not guarantee sponsorship for this specific role. Below presents additional info for your reference. (Data Powered by US Department of Labor)Distribution of Different Job Fields Receiving SponsorshipRepresents job field similar to this jobTrends of Total Sponsorships2025 (5)2024 (10)2023 (16)2022 (19)2021 (15)2020 (13)FundingCurrent StageLate StageLeadership TeamFlora X.Co-Founder and Vice PresidentCompany data provided by crunchbaseBoost Your Interview ChancesImprove Resume Match ScoreFREE?Your Score8.9Top ApplicantsMust-Have Skills for This RoleJavaSpringSQLJavaScriptAngularOptimize my ResumeGet Referral Via linkedIn FREE3× Higher Response via Email OutreachAAnna L.Senior Technical RecruiterGG H.Java Software Engineer
"""

SCHEMA:
"""
{"$schema":"http://json-schema.org/draft-04/schema#","type":"object","properties":{"personal_information":{"type":"object","properties":{"name":{"type":"string"},"email":{"type":"string"},"phone":{"type":"string"},"location":{"type":"string"},"socials":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"link":{"type":"string"}},"required":["name","link"]}]}},"required":["name","email","phone","location"]},"summary":{"type":"string"},"experiences":{"type":"array","items":[{"type":"object","properties":{"designation":{"type":"string"},"companyName":{"type":"string"},"location":{"type":"string"},"start_date":{"type":"string"},"end_date":{"type":"string"},"points":{"type":"array","items":[{"type":"string"}]}},"required":["designation","companyName","location","start_date"]}]},"education":{"type":"array","items":[{"type":"object","properties":{"institution":{"type":"string"},"degree":{"type":"string"},"location":{"type":"string"},"start_date":{"type":"string"},"end_date":{"type":"string"},"gpa":{"type":"string"}},"required":["institution","degree","location","start_date","gpa"]}]},"skills":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"data":{"type":"array","items":[{"type":"string"}]}},"required":["name","data"]}]},"projects":{"type":"array","items":[{"type":"object","properties":{"projectName":{"type":"string"},"caption":{"type":"string"},"location":{"type":"string"},"start_date":{"type":"string"},"end_date":{"type":"string"},"url":{"type":"string"},"projectDetails":{"type":"array","items":[{"type":"string"}]},"externalSources":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"link":{"type":"string"}},"required":["name","link"]}]},"technologiesUsed":{"type":"array","items":[{"type":"string"}]}},"required":["projectName","location","projectDetails"]}]},"certifications":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"issuing_organization":{"type":"string"},"issue_date":{"type":"string"},"expiration_date":{"type":"string"},"credential_id":{"type":"string"},"url":{"type":"string"}},"required":["name","issuing_organization","issue_date","expiration_date","credential_id","url"]}]},"awards":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"type":{"type":"string"},"location":{"type":"string"},"date":{"type":"string"},"description":{"type":"string"}},"required":["name","type","location","date","description"]}]},"extracurricular_achievements":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"type":{"type":"string"},"location":{"type":"string"},"date":{"type":"string"},"description":{"type":"string"}},"required":["name","type","location","date","description"]}]},"languages":{"type":"array","items":[{"type":"object","properties":{"language":{"type":"string"},"proficiency":{"type":"string"}},"required":["language","proficiency"]}]}},"required":["personal_information","education","skills","extracurricular_achievements"]}
"""

Create a tailored resume in JSON following the SCHEMA exactly.
Use only content from the RESUME_TEXT but rewrite it to match the JOB_DESCRIPTION.
Do not add extra lines or explanation.
Output only JSON.