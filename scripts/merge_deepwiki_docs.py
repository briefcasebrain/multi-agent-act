#!/usr/bin/env python3
"""
Merge DeepWiki generated documentation with existing manual documentation.
"""

import argparse
import shutil
from pathlib import Path
import re
import yaml


def merge_deepwiki_docs(deepwiki_dir: Path, existing_docs: Path, output_dir: Path):
    """
    Merge DeepWiki generated documentation with existing manual documentation.

    Args:
        deepwiki_dir: Directory containing DeepWiki generated docs
        existing_docs: Directory containing existing manual docs
        output_dir: Directory to output merged documentation
    """

    print(f"Merging DeepWiki docs from {deepwiki_dir} with existing docs from {existing_docs}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy existing docs as base
    print("Copying existing documentation...")
    for item in existing_docs.iterdir():
        if item.name not in ['deepwiki-enhanced', 'enhanced']:
            if item.is_file():
                shutil.copy2(item, output_dir / item.name)
            else:
                shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)

    # Copy DeepWiki generated content
    if deepwiki_dir.exists():
        print("Integrating DeepWiki generated content...")
        deepwiki_output = output_dir / 'ai-enhanced'
        deepwiki_output.mkdir(exist_ok=True)

        # Copy DeepWiki files
        for item in deepwiki_dir.iterdir():
            if item.is_file() and item.suffix in ['.md', '.html']:
                target = deepwiki_output / item.name

                # Process and enhance DeepWiki content
                enhanced_content = enhance_deepwiki_content(item.read_text())
                target.write_text(enhanced_content)
            elif item.is_dir():
                shutil.copytree(item, deepwiki_output / item.name, dirs_exist_ok=True)

    # Create integrated navigation
    create_integrated_navigation(output_dir)

    print(f"✅ Documentation merge completed. Output in {output_dir}")


def enhance_deepwiki_content(content: str) -> str:
    """
    Enhance DeepWiki generated content with Jekyll front matter and formatting.

    Args:
        content: Raw DeepWiki content

    Returns:
        Enhanced content with Jekyll formatting
    """

    # Add Jekyll front matter if not present
    if not content.startswith('---'):
        title = extract_title_from_content(content)
        front_matter = f"""---
layout: default
title: {title}
parent: AI-Enhanced Documentation
nav_order: 100
---

"""
        content = front_matter + content

    # Enhance code blocks with language hints
    content = re.sub(r'```(\n)', r'```python\1', content)

    # Add table of contents for long documents
    if content.count('\n') > 50:
        toc = """
## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

"""
        # Insert TOC after front matter and title
        content = re.sub(r'(---\n.*?---\n)(.*?)(#[^#])', r'\1\2' + toc + r'\3', content, flags=re.DOTALL)

    return content


def extract_title_from_content(content: str) -> str:
    """
    Extract title from content, defaulting to a reasonable title.

    Args:
        content: Document content

    Returns:
        Extracted or generated title
    """

    # Try to find first heading
    heading_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if heading_match:
        return heading_match.group(1)

    # Try to find title in file content
    title_match = re.search(r'title:\s*(.+)', content)
    if title_match:
        return title_match.group(1)

    return "AI-Enhanced Analysis"


def create_integrated_navigation(output_dir: Path):
    """
    Create integrated navigation structure including AI-enhanced content.

    Args:
        output_dir: Output directory containing merged docs
    """

    # Create AI-enhanced section index
    ai_enhanced_dir = output_dir / 'ai-enhanced'
    if ai_enhanced_dir.exists():
        ai_index_content = """---
layout: default
title: AI-Enhanced Documentation
nav_order: 7
has_children: true
---

# AI-Enhanced Documentation
{: .no_toc }

Documentation enhanced with AI analysis and insights powered by DeepWiki.

## Available AI Analysis

"""

        # List all AI-enhanced documents
        for item in ai_enhanced_dir.iterdir():
            if item.is_file() and item.suffix in ['.md', '.html']:
                title = extract_title_from_content(item.read_text())
                # Convert filename to readable format
                link_name = item.stem.replace('-', ' ').replace('_', ' ').title()
                ai_index_content += f"- [{link_name}]({item.name}) - {title}\n"

        ai_index_content += """
## About AI-Enhanced Documentation

This documentation section is automatically generated using DeepWiki's AI analysis capabilities, providing:

- **Code Architecture Analysis**: Deep insights into system design patterns
- **Dependency Mapping**: Comprehensive dependency relationships
- **Usage Pattern Detection**: Automatically identified best practices
- **Performance Insights**: AI-detected optimization opportunities
- **Documentation Gaps**: Areas where additional documentation would be valuable

The AI analysis complements the manual documentation to provide comprehensive coverage of the library's capabilities and usage patterns.

---
*AI analysis updated automatically when code changes are detected.*
"""

        (ai_enhanced_dir / 'index.md').write_text(ai_index_content)

    # Update main configuration to include AI-enhanced section
    config_file = output_dir / '_config.yml'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Add AI-enhanced navigation if not present
        if 'aux_links' not in config:
            config['aux_links'] = {}

        config['aux_links']["AI Analysis"] = ["/ai-enhanced/"]

        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


def main():
    """Main function for command-line usage."""

    parser = argparse.ArgumentParser(
        description="Merge DeepWiki generated documentation with existing docs"
    )

    parser.add_argument(
        '--deepwiki-dir',
        type=Path,
        required=True,
        help="Directory containing DeepWiki generated documentation"
    )

    parser.add_argument(
        '--existing-docs',
        type=Path,
        required=True,
        help="Directory containing existing manual documentation"
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help="Directory to output merged documentation"
    )

    args = parser.parse_args()

    # Validate input directories
    if not args.existing_docs.exists():
        raise FileNotFoundError(f"Existing docs directory not found: {args.existing_docs}")

    if not args.deepwiki_dir.exists():
        print(f"⚠️  DeepWiki directory not found: {args.deepwiki_dir}")
        print("Proceeding with existing docs only...")

    merge_deepwiki_docs(args.deepwiki_dir, args.existing_docs, args.output_dir)


if __name__ == "__main__":
    main()