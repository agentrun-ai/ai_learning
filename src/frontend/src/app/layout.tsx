import type { Metadata } from "next";
import { ThemeProvider } from "../hooks/useTheme";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI搜学助手",
  description: "AI驱动的智能学习助手",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang='zh-CN' suppressHydrationWarning>
      <head>
        {/* 防止主题闪烁的内联脚本 */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var mode = localStorage.getItem('theme-mode');
                  var theme = mode;
                  if (mode === 'auto' || !mode) {
                    theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
                  }
                  document.documentElement.classList.add(theme);
                } catch (e) {
                  document.documentElement.classList.add('dark');
                }
              })();
            `,
          }}
        />
      </head>
      <body className='antialiased'>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
