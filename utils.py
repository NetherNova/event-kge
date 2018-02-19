from urllib.parse import quote_plus


def url_parse(url):
    """Safe encode url by replacing special characters(excluding the safe symbols) with '+'"""
    return quote_plus(url, safe=';/?:@&=+$#')
