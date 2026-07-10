import "./globals.css";

export const metadata = {
  title: "Pronunciation Check",
  description: "Upload 30-45s of English speech and get word-level pronunciation feedback.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
