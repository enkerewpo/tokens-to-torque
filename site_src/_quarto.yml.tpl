project:
  type: website
  output-dir: _site
  render:
    - "*.md"
    - "days/*.md"
    - "appendix/*.md"

website:
  title: "tokens → torque"
  description: "An embodied-AI stack from scratch — 72 days, 2 hours a day, on a Jetson."
  site-url: https://enkerewpo.github.io/tokens-to-torque/
  repo-url: https://github.com/enkerewpo/tokens-to-torque
  repo-actions: [source]
  search:
    location: navbar
    type: overlay
  navbar:
    left:
      - text: 课表
        href: roadmap.md
      - text: 环境与安全
        href: setup.md
      - text: 资源
        href: resources.md
      - text: 用 agent 跟做
        href: agents.md
    right:
      - icon: github
        href: https://github.com/enkerewpo/tokens-to-torque
        aria-label: GitHub
  sidebar:
    style: floating
    collapse-level: 2
    contents:
      - text: 首页
        href: index.md
      - section: Days
        contents:
__DAYS__
      - section: 附录
        contents:
__APPENDIX__
  page-footer:
    left: "MIT · wheatfox"
    right: "用 [Quarto](https://quarto.org) 构建"

format:
  html:
    theme:
      light: [cosmo, assets/custom.scss]
      dark: [darkly, assets/custom-dark.scss]
    toc: true
    toc-depth: 3
    toc-title: 目录
    toc-expand: 2
    number-sections: false
    anchor-sections: true
    code-copy: true
    code-overflow: scroll
    highlight-style: github
    html-math-method: katex
    link-external-newwindow: true
    lang: zh-Hans

language:
  section-title-footnotes: 参考文献
  toc-title-document: 目录
