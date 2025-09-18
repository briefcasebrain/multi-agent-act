/**
 * Enhanced Mermaid Initialization
 * Handles automatic diagram rendering with custom styling
 */

document.addEventListener('DOMContentLoaded', function() {
  // Enhanced Mermaid configuration
  mermaid.initialize({
    startOnLoad: false, // We'll control the initialization
    theme: 'default',
    themeVariables: {
      primaryColor: '#3b82f6',
      primaryTextColor: '#1f2937',
      primaryBorderColor: '#2563eb',
      lineColor: '#6b7280',
      secondaryColor: '#f0f9ff',
      tertiaryColor: '#dbeafe',
      background: '#ffffff',
      mainBkg: '#ffffff',
      fontFamily: 'Inter, system-ui, sans-serif'
    },
    flowchart: {
      useMaxWidth: true,
      htmlLabels: true,
      curve: 'basis'
    },
    sequence: {
      useMaxWidth: true,
      wrap: true
    },
    gantt: {
      useMaxWidth: true
    },
    securityLevel: 'loose',
    maxTextSize: 90000
  });

  // Function to render Mermaid diagrams
  function renderMermaidDiagrams() {
    console.log('Searching for Mermaid diagrams...');

    // Find all code blocks with mermaid class - try multiple selectors
    const selectors = [
      'code.language-mermaid',
      '.language-mermaid code',
      'pre code.language-mermaid',
      '.highlighter-rouge .language-mermaid',
      'pre.highlight code.language-mermaid'
    ];

    let mermaidBlocks = [];
    selectors.forEach(selector => {
      const found = document.querySelectorAll(selector);
      mermaidBlocks = mermaidBlocks.concat(Array.from(found));
    });

    console.log(`Found ${mermaidBlocks.length} Mermaid code blocks`);

    mermaidBlocks.forEach((block, index) => {
      const diagramCode = block.textContent || block.innerText;
      console.log(`Processing diagram ${index}:`, diagramCode.substring(0, 50) + '...');

      // Create a new div for the diagram
      const diagramDiv = document.createElement('div');
      diagramDiv.className = 'mermaid-diagram';
      diagramDiv.id = `mermaid-diagram-${index}`;
      diagramDiv.textContent = diagramCode; // Set the code content for mermaid to process

      // Find the container to replace (could be pre, div.highlighter-rouge, etc.)
      let container = block;
      while (container.parentElement &&
             !container.classList.contains('highlighter-rouge') &&
             container.tagName.toLowerCase() !== 'pre') {
        container = container.parentElement;
      }

      // If we found a container, replace it, otherwise replace the block itself
      const parentElement = container.parentElement || block.parentElement;
      const elementToReplace = container.parentElement ? container : block;

      parentElement.replaceChild(diagramDiv, elementToReplace);

      console.log(`Replaced element for diagram ${index}`);
    });

    // Now initialize mermaid to process all the diagram divs
    if (mermaidBlocks.length > 0) {
      console.log('Initializing Mermaid rendering...');
      try {
        mermaid.run({
          querySelector: '.mermaid-diagram'
        });
        console.log('Mermaid rendering complete');
      } catch (error) {
        console.error('Mermaid rendering error:', error);
      }
    }
  }

  // Enhanced code block processing
  function enhanceCodeBlocks() {
    const codeBlocks = document.querySelectorAll('div.highlighter-rouge');

    codeBlocks.forEach(block => {
      const pre = block.querySelector('pre');
      const code = block.querySelector('code');

      if (code && code.className) {
        // Extract language from class name
        const langMatch = code.className.match(/language-(\w+)/);
        if (langMatch) {
          const language = langMatch[1];
          block.classList.add(`language-${language}`);

          // Add copy button
          const copyButton = document.createElement('button');
          copyButton.className = 'copy-code-button';
          copyButton.innerHTML = '<i class="fas fa-copy"></i>';
          copyButton.title = 'Copy code';

          copyButton.addEventListener('click', () => {
            const codeText = code.textContent || code.innerText;
            navigator.clipboard.writeText(codeText).then(() => {
              copyButton.innerHTML = '<i class="fas fa-check"></i>';
              setTimeout(() => {
                copyButton.innerHTML = '<i class="fas fa-copy"></i>';
              }, 2000);
            });
          });

          block.style.position = 'relative';
          block.appendChild(copyButton);
        }
      }
    });
  }

  // Initialize everything
  renderMermaidDiagrams();
  enhanceCodeBlocks();

  // Re-process if new content is loaded dynamically
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.addedNodes.length > 0) {
        renderMermaidDiagrams();
        enhanceCodeBlocks();
      }
    });
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
});