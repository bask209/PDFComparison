"""
Module that runs streamlit app. Uses OpenCV, PyMuPDF, Numpy and Streamlit.

When running the app use command "streamlit run ./PDFComparison.py"

"""
import cv2
import fitz
import numpy as np
import streamlit as st

def compare_pdf_images(pdf1_path, pdf2_path, output_path):
    pdf1 = fitz.open(stream=pdf1_path.read(),filetype="pdf")
    pdf2 = fitz.open(stream=pdf2_path.read(),filetype="pdf")

    # Iterate through pages
    for page_num in range(min(pdf1.page_count, pdf2.page_count)):
        # Load page images
        page1 = pdf1.load_page(page_num)
        page2 = pdf2.load_page(page_num)
        pix1 = page1.get_pixmap()
        pix2 = page2.get_pixmap()

        # Convert images to NumPy arrays
        img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.h, pix1.w, pix1.n)
        img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.h, pix2.w, pix2.n)

        # Convert BGR to RGB color space
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Preprocess images (e.g., resize, threshold, filter)

        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # Compute absolute difference between grayscale images
        diff = cv2.absdiff(gray1, gray2)

        # Convert difference image to binary
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Highlight differences with bounding boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save comparison results
        #output_file = f"output_{page_num}.png"
        #cv2.imwrite(output_file, img1)

    # Close PDFs
    pdf1.close()
    pdf2.close()

    # Save combined output image
    cv2.imwrite(output_path, img1)

# Streamlit app
def app():
    st.title("PDF Comparison App")
    st.write("Upload two PDF files for comparison")

    # File uploader for PDF 1
    file1 = st.file_uploader("Upload PDF 1", type=["pdf"])

    # File uploader for PDF 2
    file2 = st.file_uploader("Upload PDF 2", type=["pdf"])

    # Compare button
    if st.button("Compare"):
        if file1 is not None and file2 is not None:
            output_path = f"comparison_{file1.name}.png"
            compare_pdf_images(file1, file2, output_path)
            st.write("Comparison complete!")
            st.image(output_path)

if __name__ == '__main__':
    app()