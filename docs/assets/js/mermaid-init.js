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
    // Find all code blocks with mermaid class
    const mermaidBlocks = document.querySelectorAll('code.language-mermaid, .language-mermaid');

    mermaidBlocks.forEach((block, index) => {
      const diagramCode = block.textContent || block.innerText;

      // Create a new div for the diagram
      const diagramDiv = document.createElement('div');
      diagramDiv.className = 'mermaid';
      diagramDiv.id = `mermaid-diagram-${index}`;

      // Replace the code block with the diagram div
      const parent = block.parentElement;
      if (parent.tagName.toLowerCase() === 'pre') {
        parent.parentNode.replaceChild(diagramDiv, parent);
      } else {
        block.parentNode.replaceChild(diagramDiv, block);
      }

      // Render the diagram
      try {
        mermaid.render(`mermaid-svg-${index}`, diagramCode, (svgCode) => {
          diagramDiv.innerHTML = svgCode;
        });
      } catch (error) {
        console.error('Mermaid rendering error:', error);
        diagramDiv.innerHTML = `<p style="color: red;">Error rendering diagram: ${error.message}</p>`;
      }
    });
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