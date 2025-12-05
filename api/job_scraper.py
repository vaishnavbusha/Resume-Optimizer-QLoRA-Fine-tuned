"""
Job Description Scraper Module.
Handles fetching job descriptions from URLs using web scraping.
"""

import re
from typing import Optional
from urllib.parse import urlparse

# Web scraping imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class JobScraper:
    """
    Scraper for extracting job descriptions from job posting URLs.
    Supports common job boards like LinkedIn, Indeed, Glassdoor, etc.
    """
    
    # Common user agent for requests
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    
    # Timeout for HTTP requests
    REQUEST_TIMEOUT = 15
    
    def __init__(self):
        """Initialize the job scraper."""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if not REQUESTS_AVAILABLE:
            print("⚠️ requests not installed. URL scraping will not work.")
        if not BS4_AVAILABLE:
            print("⚠️ beautifulsoup4 not installed. HTML parsing will not work.")
    
    def scrape(self, url: str) -> str:
        """
        Scrape job description from a URL.
        
        Args:
            url: URL of the job posting
        
        Returns:
            Extracted job description text
        
        Raises:
            ValueError: If scraping fails or dependencies are missing
        """
        if not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
            raise ValueError(
                "Web scraping dependencies not available. "
                "Install with: pip install requests beautifulsoup4"
            )
        
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        
        try:
            # Fetch the page
            headers = {"User-Agent": self.USER_AGENT}
            response = requests.get(url, headers=headers, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Try to extract job description based on common patterns
            job_text = self._extract_job_description(soup, url)
            
            if not job_text or len(job_text) < 100:
                raise ValueError(
                    "Could not extract meaningful job description. "
                    "Please copy and paste the job description text directly."
                )
            
            return job_text
            
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch URL: {str(e)}")
    
    def _extract_job_description(self, soup: BeautifulSoup, url: str) -> str:
        """
        Extract job description from parsed HTML.
        Uses various strategies based on common job board structures.
        """
        domain = urlparse(url).netloc.lower()
        
        # LinkedIn
        if "linkedin.com" in domain:
            return self._extract_linkedin(soup)
        
        # Indeed
        elif "indeed.com" in domain:
            return self._extract_indeed(soup)
        
        # Glassdoor
        elif "glassdoor.com" in domain:
            return self._extract_glassdoor(soup)
        
        # Generic extraction
        else:
            return self._extract_generic(soup)
    
    def _extract_linkedin(self, soup: BeautifulSoup) -> str:
        """Extract job description from LinkedIn."""
        # LinkedIn job description selectors
        selectors = [
            ".description__text",
            ".show-more-less-html__markup",
            "[class*='job-description']",
            "[class*='description']",
            "article",
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = self._clean_html_text(element)
                if len(text) > 100:
                    return text
        
        return self._extract_generic(soup)
    
    def _extract_indeed(self, soup: BeautifulSoup) -> str:
        """Extract job description from Indeed."""
        selectors = [
            "#jobDescriptionText",
            ".jobsearch-jobDescriptionText",
            "[class*='job-description']",
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = self._clean_html_text(element)
                if len(text) > 100:
                    return text
        
        return self._extract_generic(soup)
    
    def _extract_glassdoor(self, soup: BeautifulSoup) -> str:
        """Extract job description from Glassdoor."""
        selectors = [
            ".desc",
            "[class*='jobDescription']",
            "[class*='description']",
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = self._clean_html_text(element)
                if len(text) > 100:
                    return text
        
        return self._extract_generic(soup)
    
    def _extract_generic(self, soup: BeautifulSoup) -> str:
        """
        Generic extraction strategy.
        Looks for common patterns in job posting pages.
        """
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        
        # Try common job description selectors
        selectors = [
            "[class*='job-description']",
            "[class*='jobDescription']",
            "[class*='description']",
            "[id*='job-description']",
            "[id*='description']",
            "article",
            "main",
            ".content",
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                text = self._clean_html_text(element)
                if len(text) > 200:
                    return text
        
        # Fallback: get all paragraph text
        paragraphs = soup.find_all("p")
        text_parts = [self._clean_html_text(p) for p in paragraphs]
        text = "\n\n".join(t for t in text_parts if len(t) > 20)
        
        return text
    
    def _clean_html_text(self, element) -> str:
        """Clean text extracted from HTML element."""
        # Get text content
        text = element.get_text(separator="\n", strip=True)
        
        # Replace multiple whitespace with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()


# Singleton instance
job_scraper = JobScraper()
