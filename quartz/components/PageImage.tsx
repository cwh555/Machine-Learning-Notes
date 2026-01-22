import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { joinSegments } from "../util/path"

const PageImage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
  const imageName = fileData.frontmatter?.image
  if (!imageName) return null

  // --- 修正邏輯開始 ---
  // GitHub Pages 專案名稱 (從你的截圖看是這個)
  // 如果你的倉庫名稱不一樣，請修改這裡！
  const repoName = "Machine-Learning-Notes" 
  
  // 我們不再依賴 pathToRoot 的相對路徑，因為 GitHub Pages 的 trailing slash 會搞亂它
  // 我們直接建立絕對路徑: /Machine-Learning-Notes/static/images/xxx.jpg
  // 這樣無論你在哪一層資料夾，瀏覽器都能精準找到位置
  const imagePath = `/${repoName}/static/images/${imageName}`
  // --- 修正邏輯結束 ---

  return (
    <div className={`page-image ${displayClass ?? ""}`}>
      <img 
        src={imagePath} 
        alt={fileData.frontmatter?.title || "Featured Image"} 
        // 這裡加一個 onError 事件，如果圖片還是掛掉，方便我們除錯
        onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.style.display = 'none'; // 圖片破圖時先隱藏，避免醜醜的圖標
            console.error(`圖片載入失敗: ${target.src}`);
            // 如果你想在畫面上直接看到錯誤路徑，可以把下面這行取消註解
            // target.parentElement!.innerText = `載入失敗: ${target.src}`;
        }}
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