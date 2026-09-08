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
  favicon: assets/favicon.svg
  search:
    location: navbar
    type: overlay
  navbar:
    # 左上角是「方标 + 站名」，不再用整幅字标：首页的大标题和 hero 图已经写了
    # 一遍站名，字标是第三遍。方标和 favicon 同形，标签页和导航栏能对上。
    logo: assets/mark.svg
    logo-alt: "tokens to torque"
    title: "tokens → torque"
    left:
      - text: 教程
        menu:
__DAYS_MENU__
      - text: 课表
        href: roadmap.md
      - text: 附录
        menu:
__APPENDIX_MENU__
      - text: 环境搭建
        href: setup.md
      - text: 精选材料
        href: resources.md
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
      - section: 教程
        contents:
__DAYS__
      - section: 附录
        contents:
__APPENDIX__
  # 每页底部挂 GitHub Discussions 评论（giscus）。ID 是 GraphQL 查出来的，
  # 分类用 Announcements：只有维护者能在那里开帖，giscus 才不会匹配到路人开的帖。
  comments:
    giscus:
      repo: enkerewpo/tokens-to-torque
      repo-id: R_kgDOUNy35Q
      category: Announcements
      category-id: DIC_kwDOUNy35c4DFIB2
      mapping: pathname
      reactions-enabled: true
      input-position: top
      loading: lazy
      language: zh-CN
      theme:
        light: light
        dark: dark_dimmed
  page-footer:
    left: "基于 [MIT 许可证](https://github.com/enkerewpo/tokens-to-torque/blob/main/LICENSE)发布 · 作者 [wheatfox](https://www.oscommunity.cn/)"
    right: "用 [Quarto](https://quarto.org) 构建，托管在 GitHub Pages"

format:
  html:
    lightbox: auto
    from: markdown+autolink_bare_uris+emoji
    theme:
      light: [cosmo, assets/custom.scss]
      dark: [darkly, assets/custom-dark.scss]
    # 主题三态（跟随系统 / 浅色 / 深色）。Quarto 自己默认写死浅色，见 assets/theme-head.html
    include-in-header:
      - assets/fonts.html
      - assets/theme-head.html
    include-after-body:
      - assets/theme-body.html
      - assets/figzoom.html
    toc: true
    toc-depth: 3
    toc-title: 目录
    toc-expand: 2
    number-sections: false
    anchor-sections: true
    code-copy: true
    code-overflow: wrap          # 长命令折行，不横着拖
    highlight-style: github
    html-math-method: katex
    link-external-newwindow: true
    lang: zh-Hans

language:
  section-title-footnotes: 参考文献
  toc-title-document: 目录
