// quartz/components/pages/HistoryNotesList.tsx
import { QuartzComponent, QuartzComponentProps } from "../types"

const HistoryNotesList: QuartzComponent = (props: QuartzComponentProps) => {
  // 只在首頁 (slug === "index") 顯示
  if (props.fileData?.slug !== "index") {
    return null
  }

  const pages = props.allFiles ?? []

  // 過濾 content/History 底下的檔案
  const historyNotes = pages
    .filter((p: any) => {
      const filePath = (p?.filePath ?? "").replace(/^\/+/,'')
      return filePath.startsWith("History/") || filePath.startsWith("content/History/")
    })
    .filter((p: any) => (p?.slug ?? "") !== "index")

  if (!historyNotes.length) {
    return <p>No history notes found.</p>
  }

  // 可依日期排序（有 date 用 date，沒有用 gitCommitDate，沒有則放最後）
  historyNotes.sort((a: any, b: any) => {
    const aDate = a?.date ?? a?.gitCommitDate ?? ""
    const bDate = b?.date ?? b?.gitCommitDate ?? ""
    const aTs = Number.isFinite(Date.parse(String(aDate))) ? Date.parse(String(aDate)) : 0
    const bTs = Number.isFinite(Date.parse(String(bDate))) ? Date.parse(String(bDate)) : 0
    return bTs - aTs
  })

  return (
    <div>
      <h2>History</h2>
      <ul>
        {historyNotes.map((note: any) => {
          const filePath = note?.filePath ?? ""
          const slug = note?.slug ?? ""
          const url = filePath ? `/${filePath.replace(/\.md$/, "").replace(/^\/+/,'')}` : `/${slug}`
          const title = note?.title ?? slug ?? url
          return (
            <li key={url}>
              <a href={url}>{title}</a>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default HistoryNotesList
