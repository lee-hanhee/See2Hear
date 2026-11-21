import CoreGraphics

struct CameraFrame {
    /// The camera producer supplies an upright, unmirrored RGB image.
    let image: CGImage

    var widthPixels: Int { image.width }
    var heightPixels: Int { image.height }
}
