import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, joinSegments } from "../util/path" // 1. 引入這兩個官方工具

const PageImage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
  const imageName = fileData.frontmatter?.image

  if (!imageName) {
    return null
  }

  // 2. 使用 joinSegments 自動處理路徑拼接
  // fileData.slug! 代表當前頁面的路徑
  // pathToRoot 會算出相對路徑 (例如 "." 或 "../..")
  // joinSegments 會自動補上斜線，變成 "./static/images/xxx.jpg" 或 "../../static/images/xxx.jpg"
  const baseDir = pathToRoot(fileData.slug!)
  const imagePath = joinSegments(baseDir, "static/images", imageName)

  return (
    <div className={`page-image ${displayClass ?? ""}`}>
      <img 
        src={imagePath} 
        alt={fileData.frontmatter?.title || "Featured Image"} 
        style={{
          width: "100%",
          borderRadius: "8px",
          marginBottom: "1rem",
          height: "auto",
          display: "block"
        }}
      />
    </div>
  )
}

export default (() => PageImage) satisfies QuartzComponentConstructor