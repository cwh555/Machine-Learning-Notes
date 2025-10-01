import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/footer.scss"
import { version } from "../../package.json"
import { i18n } from "../i18n"

interface LinkItem {
  text: string
  url?: string
}

interface Options {
  links?: Record<string, LinkItem>   // 有 url 的才生成 <a>
  text?: Record<string, LinkItem>    // 純文字
}

export default ((opts?: Options) => {
  const Footer: QuartzComponent = ({ displayClass, cfg }: QuartzComponentProps) => {
    const year = new Date().getFullYear()
    const links = opts?.links ?? {}
    const texts = opts?.text ?? {}

    return (
      <footer class={`${displayClass ?? ""}`}>
        <p>
          {i18n(cfg.locale).components.footer.createdWith}{" "}
          <a href="https://quartz.jzhao.xyz/">Quartz v{version}</a> © {year}
        </p>
        <ul>
          {Object.entries(links).map(([key, item]) => (
            <li>
              <a href={item.url}>{item.text}</a>
            </li>
          ))}
          {Object.entries(texts).map(([key, item]) => (
            <li>{item.text}</li>
          ))}
        </ul>
      </footer>
    )
  }

  Footer.css = style
  return Footer
}) satisfies QuartzComponentConstructor