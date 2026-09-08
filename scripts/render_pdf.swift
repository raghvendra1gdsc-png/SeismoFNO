import Foundation
import PDFKit
import AppKit

guard CommandLine.arguments.count >= 3 else {
    print("Usage: swift render_pdf.swift <input.pdf> <output_pattern> [dpi]")
    exit(1)
}

let pdfPath = CommandLine.arguments[1]
let outPattern = CommandLine.arguments[2]
let dpi: CGFloat = CommandLine.arguments.count >= 4 ? CGFloat(Double(CommandLine.arguments[3]) ?? 150.0) : 150.0

let fullUrl = URL(fileURLWithPath: pdfPath)
guard let doc = PDFDocument(url: fullUrl) else {
    print("Could not open PDF: \(pdfPath)")
    exit(1)
}

let scale = dpi / 72.0

for i in 0..<doc.pageCount {
    guard let page = doc.page(at: i) else { continue }
    let bounds = page.bounds(for: .mediaBox)
    let width = Int(bounds.width * scale)
    let height = Int(bounds.height * scale)
    
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: width,
        pixelsHigh: height,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else {
        continue
    }
    
    NSGraphicsContext.saveGraphicsState()
    guard let context = NSGraphicsContext(bitmapImageRep: rep) else { continue }
    NSGraphicsContext.current = context
    
    context.cgContext.setFillColor(NSColor.white.cgColor)
    context.cgContext.fill(CGRect(x: 0, y: 0, width: width, height: height))
    
    context.cgContext.scaleBy(x: scale, y: scale)
    page.draw(with: .mediaBox, to: context.cgContext)
    
    NSGraphicsContext.restoreGraphicsState()
    
    guard let pngData = rep.representation(using: .png, properties: [:]) else { continue }
    
    let outPath = String(format: outPattern, i + 1)
    try? pngData.write(to: URL(fileURLWithPath: outPath))
    print("Wrote page \(i + 1) to \(outPath)")
}
