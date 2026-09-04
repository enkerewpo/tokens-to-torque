project:
  type: website
  output-dir: _site
  # 只在 SCSS 的 url() 里出现的资源，Quarto 不会自动收集，要显式声明
  resources:
    - assets/*.svg
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
  favicon: assets/favicon.svg
  search:
    location: navbar
    type: overlay
  navbar:
    logo: assets/logo-light.svg
    logo-alt: "tokens to torque"
    title: false
    left:
      - text: 课表
        href: roadmap.md
      - text: 环境
        href: setup.md
      - text: 资源
        href: resources.md
      - text: Agent
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
    from: markdown+autolink_bare_uris+emoji
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
