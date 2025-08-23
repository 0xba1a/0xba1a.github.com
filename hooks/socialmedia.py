from textwrap import dedent
import urllib.parse
import re

linked_intent = "https://www.linkedin.com/sharing/share-offsite"
x_intent = "https://twitter.com/intent/tweet"
fb_sharer = "https://www.facebook.com/sharer/sharer.php"
include = re.compile(r"blog/[1-9].*")

def on_page_markdown(markdown, **kwargs):
    page = kwargs['page']
    config = kwargs['config']
    if not include.match(page.url):
        return markdown

    page_url = urllib.parse.quote(config.site_url+page.url, safe=':/?&=')
    # page_title = urllib.parse.quote(page.title+'\n')

    return markdown + dedent(f"""<div class="flex">
        <script src="https://platform.linkedin.com/in.js" type="text/javascript">lang: en_US</script>
        <script type="IN/Share" data-url="{page_url}"></script>
    </div>""")

        # <script type="text/javascript" async src="https://platform.twitter.com/widgets.js"></script>
        # <a class="twitter-share-button" href="https://twitter.com/intent/tweet" data-size="small" data-text="{page.title} by @0xba1a" data-url="{page_url}" data-hashtags="eastrivervillage" data-related="twitterapi,twitter">:simple-x:</a>

