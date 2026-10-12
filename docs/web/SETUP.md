# Publish "Latitude and Doubt" on GitHub Pages

## Files
index.html (main page) · Latitude_Doubt.pdf · bie.html (E. coli demo) · comic.html (earlier comic guide)
llms.txt · robots.txt · sitemap.xml · 404.html · .nojekyll

## 1. Upload
Put every file in the ROOT of https://github.com/4t1a/Latitude-Doubt (next to your README, code/, data/).
Web: Add file > Upload files > drag the files > Commit.
Terminal:
    git clone https://github.com/4t1a/Latitude-Doubt && cd Latitude-Doubt
    cp -r /path/to/Latitude-Doubt/. .        # includes .nojekyll
    git add . && git commit -m "Add explainer site" && git push

## 2. Turn on Pages
Settings > Pages > Source: Deploy from a branch > Branch: main, folder: / (root) > Save.
Live in about a minute at https://4t1a.github.io/Latitude-Doubt/

## 3. Check
- https://4t1a.github.io/Latitude-Doubt/Latitude_Doubt.pdf opens the PDF
- https://4t1a.github.io/Latitude-Doubt/llms.txt and /sitemap.xml load
- Rich Results Test (search.google.com/test/rich-results): paste the page URL, expect FAQ + Article data
- In the repo, click the gear next to "About": set the website to the Pages URL and add topics:
  information-theory, entropy, genetic-code, steganography, semantic-information, preprint

## 4. Search engines
Google Search Console: add property https://4t1a.github.io/Latitude-Doubt/ (URL prefix),
verify (HTML tag or file), then Sitemaps > submit sitemap.xml and request indexing of the home page.
Bing Webmaster Tools: import from Search Console. This also feeds Bing-based AI answers.

## 5. Google Scholar
Scholar cannot be told to index a page; it crawls on its own. To speed it up:
1. Keep the PDF at its stable URL above. index.html already carries the citation_* tags.
2. scholar.google.com > My profile > + (Add) > Add article manually: title, author, date, and the DOI link.
3. In 2 to 6 weeks search the exact title. If Zenodo and your copy both appear, merge them in your profile.
4. Put the DOI and the Pages link in the repo README, CITATION.cff and the Zenodo record (related identifiers).

## 6. After changes
Edit, commit, push. Update <lastmod> in sitemap.xml when you change content.
