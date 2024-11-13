from pytube import Search

# 検索キーワードを指定
search_keyword = '`切り抜き'

# Searchオブジェクトを作成
search = Search(search_keyword)
# 検索結果を取得
results = search.results
# search.get_next_results()
# search.get_next_results()

print(len(results))

# 各動画のタイトルとURLを表示
for video in results:
    print(f'Title: {video.title}')
    print(f'URL: {video.watch_url}')
    print('---')
