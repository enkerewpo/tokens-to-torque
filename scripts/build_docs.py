#!/usr/bin/env python3
"""把仓库里的 markdown 组装成 Quarto 站点源码（site_src/），并生成侧栏。

内容的唯一来源是根目录的 README/ROADMAP/SETUP/RESOURCES/AGENTS 和每天的
days/dayNN_*/README.md。site_src/ 下的 .md 全是生成物（已 gitignore），
手改会被下次构建覆盖。新增一天只要建 days/dayNN_topic/README.md。

源码用 GitHub alert 语法（`> [!NOTE]`），这样直接在 GitHub 上看仓库也正常；
这里把它转成 Quarto 的 callout。
"""
import pathlib, re, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "site_src"
GH = "https://github.com/enkerewpo/tokens-to-torque/blob/main"

TOP = {"README.md": "index.md", "ROADMAP.md": "roadmap.md", "SETUP.md": "setup.md",
       "RESOURCES.md": "resources.md", "AGENTS.md": "agents.md"}

ALERT = {"NOTE": "note", "TIP": "tip", "IMPORTANT": "important",
         "WARNING": "warning", "CAUTION": "caution"}


def alerts_to_callouts(text: str) -> str:
    """> [!WARNING] / > **标题** / > 正文   ->   ::: {.callout-warning title="标题"}"""
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        m = re.match(r"^>\s*\[!(\w+)\]\s*$", lines[i])
        if not m:
            out.append(lines[i]); i += 1; continue
        kind = ALERT.get(m.group(1).upper(), "note")
        i += 1
        body = []
        while i < len(lines) and lines[i].startswith(">"):
            body.append(re.sub(r"^>\s?", "", lines[i]))
            i += 1
        title = ""
        if body and re.fullmatch(r"\*\*(.+)\*\*", body[0].strip()):
            title = re.fullmatch(r"\*\*(.+)\*\*", body[0].strip()).group(1)
            body = body[1:]
            while body and not body[0].strip():
                body = body[1:]
        head = f'::: {{.callout-{kind}'
        head += f' title="{title}"}}' if title else "}"
        out.append(head)
        out.extend(body)
        out.append(":::")
    return "\n".join(out)


def rewrite_links(text: str, in_day: bool, in_sub: bool = None) -> str:
    """in_day：文件在 days/ 下；in_sub：文件在任意一级子目录下（days/ 或 appendix/）。"""
    if in_sub is None:
        in_sub = in_day
    up = "../" if in_sub else ""
    text = re.sub(r"\]\((?:\.\./)*days/day(\d\d)_[a-z0-9-]+/(#[^)]*)?\)",
                  lambda m: f"]({'' if in_day else up + 'days/'}day{m.group(1)}.md{m.group(2) or ''})", text)
    text = re.sub(r"\]\((?:\.\./)*appendix/([a-z0-9-]+)\.md(#[^)]*)?\)",
                  lambda m: f"]({up}appendix/{m.group(1)}.md{m.group(2) or ''})", text)
    for src, dst in TOP.items():
        text = re.sub(re.escape(f"]({src}") + r"(#[^)]*)?\)",
                      lambda m, d=dst: f"]({up}{d}{m.group(1) or ''})", text)
    text = re.sub(r"\]\((?:\.\./)*(common/[^)]*|templates/[^)]*|scripts/[^)]*|LICENSE)\)",
                  lambda m: f"]({GH}/{m.group(1)})", text)
    # results/ 下的图片复制进站点本地引用；其余 code/ results/ 文件链到 GitHub
    # 交互式小组件（附录 D 的注意力热力图）：源文件里只放一个占位符，
    # 真正的 HTML 由 site_src/assets/ 下的文件在这里插进去。
    if "<!-- WIDGET: attn -->" in text:
        w = (SRC / "assets" / "attn-widget.html").read_text(encoding="utf-8")
        # 数据直接嵌进页面：靠 fetch 取的话，Quarto 扫不到这个引用、不会把
        # 文件拷进产物，本地用 file:// 打开也会被浏览器拦掉。
        w = w.replace("<!-- ATTN-DATA -->",
                      (SRC / "assets" / "attn-demo.json").read_text(encoding="utf-8"))
        # 用 ```{=html} 围栏包住：直接贴 HTML 的话，缩进 4 空格的那几行会被
        # markdown 当成代码块，标签就原样显示出来了。
        text = text.replace("<!-- WIDGET: attn -->", "```{=html}\n" + w + "\n```")

    # 插图直接内联进页面，不走 <img>：<img> 里的 SVG 是独立文档，拿不到页面
    # 用 CDN 加载的字体，图里的中文会掉回系统字体，和正文对不上。内联之后它
    # 用的就是页面的字体和字号，缩放也跟着正文走。外面套一个 .lightbox 的链接
    # 保留点开放大。
    def _inline_svg(m):
        rel, attrs = m.group(1), m.group(2)
        f = SRC / rel
        if not f.exists():
            return m.group(0)
        svg = f.read_text(encoding="utf-8").strip()
        classes = " ".join(re.findall(r"\.([a-z][a-z0-9-]*)", attrs))
        svg = re.sub(r'\s(?:width|height)="[\d.]+"', "", svg, count=2)
        svg = svg.replace("<svg ", f'<svg class="{classes}" ', 1)
        # 不套 <a>：Quarto 自带的 glightbox 会抢过去，把 .svg 当图片打开，
        # 那样又回到系统字体了。放大交给 assets/figzoom.html 里的浮层。
        return "```{=html}\n" + svg + "\n```"

    text = re.sub(r"!\[\]\((?:\.\./)*site_src/(assets/[^)]+\.svg)\)\{([^}]*)\}",
                  _inline_svg, text)

    # 图片资源在 site_src/assets/：源码里可能写成 site_src/… 或 ../site_src/…，
    # 统一改成相对站点的 assets/…（子目录页面要加 ../）
    text = re.sub(r"\]\((?:\.\./)*site_src/(assets/[^)]*)\)",
                  lambda m: f"]({up}{m.group(1)})", text)
    text = re.sub(r"\]\((results/[^)]*\.(?:png|jpg|jpeg|svg|gif))\)",
                  lambda m: f"](assets/DAYDIR/{m.group(1)})", text)
    text = re.sub(r"\]\((code/[^)]*|results/[^)]*)\)",
                  lambda m: f"]({GH}/DAYDIR/{m.group(1)})", text)
    return text


def write(path: pathlib.Path, content: str, written: set) -> None:
    """内容没变就不写，避免 preview 反复重渲染。"""
    written.add(path)
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def frontmatter(title: str) -> str:
    return f'---\ntitle: "{title}"\n---\n\n'


def strip_h1(text: str):
    """Quarto 用 frontmatter 的 title 生成标题，正文里的 H1 要去掉，否则重复。"""
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if not m:
        return None, text
    return m.group(1).strip(), text[:m.start()] + text[m.end():].lstrip("\n")


def process(path: pathlib.Path, in_day: bool, daydir: str = "", in_sub: bool = None) -> tuple[str, str]:
    body = alerts_to_callouts(rewrite_links(path.read_text(), in_day, in_sub))
    if daydir:
        body = body.replace("DAYDIR", daydir)
    title, body = strip_h1(body)
    return title or path.stem, body


def main():
    # 注意：不要"先删光再重写"。quarto preview 在监听这些文件，
    # 删除到重写之间的空窗期会让它扫到文件消失并卡在 Render Error。
    # 改成先写新内容、最后再清理这一轮没生成的旧文件。
    (SRC / "days").mkdir(parents=True, exist_ok=True)
    (SRC / "appendix").mkdir(parents=True, exist_ok=True)
    written: set[pathlib.Path] = set()

    for src, dst in TOP.items():
        f = ROOT / src
        if not f.exists():
            continue
        title, body = process(f, in_day=False)
        # 首页的居中横幅在 Quarto 里由 frontmatter 接管，去掉原来的 <div>
        if dst == "index.md":
            body = re.sub(r'<div align="center">\n(.*?)\n</div>', r"\1", body, flags=re.S)
        write(SRC / dst, frontmatter(title) + body, written)

    days = []
    for d in sorted((ROOT / "days").iterdir()):
        readme = d / "README.md"
        m = re.match(r"day(\d\d)_(.+)", d.name)
        if not d.is_dir() or not readme.exists() or not m:
            continue
        num = m.group(1)
        title, body = process(readme, in_day=True, daydir=d.name)
        write(SRC / "days" / f"day{num}.md", frontmatter(title) + body, written)
        days.append((num, title))
        res = d / "results"
        if res.is_dir():
            dst = SRC / "days" / "assets" / d.name / "results"
            dst.mkdir(parents=True, exist_ok=True)
            for img in res.iterdir():
                if img.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg", ".gif"):
                    shutil.copy2(img, dst / img.name)

    apps = []
    for f in sorted((ROOT / "appendix").glob("*.md")):
        title, body = process(f, in_day=False, in_sub=True)
        write(SRC / "appendix" / f.name, frontmatter(title) + body, written)
        apps.append((f.stem, title))
    apps.sort(key=lambda a: a[1])          # 按标题「附录 A/B/C」排序，而不是按文件名

    tpl = (SRC / "_quarto.yml.tpl").read_text()
    entries = "\n".join(f"          - text: \"{t}\"\n            href: days/day{n}.md"
                        for n, t in days) or "          - text: (还没开始)\n            href: index.md"
    app_entries = "\n".join(f"          - text: \"{t}\"\n            href: appendix/{n}.md" for n, t in apps)
    (SRC / "_quarto.yml").write_text(tpl.replace("__DAYS__", entries).replace("__APPENDIX__", app_entries))

    for stale in list(SRC.glob("*.md")) + list((SRC / "days").glob("*.md")) + list((SRC / "appendix").glob("*.md")):
        if stale not in written:
            stale.unlink()

    print(f"site_src/ 生成完毕：{len(TOP)} 个顶层页 + {len(days)} 天 + {len(apps)} 个附录")
    for n, t in days:
        print(f"  day{n}  {t}")


if __name__ == "__main__":
    sys.exit(main())
