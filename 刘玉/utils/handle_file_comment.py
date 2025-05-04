# 定义一个类来表示每条评论信息
class BookComment:
    def __init__(self, book_name, author_id, star, time, likenum, body):
        self.author_id = author_id
        self.star = star
        self.time = time
        self.likenum = likenum
        self.body = body
        self.book_name = book_name

    def __repr__(self):
        return f"BookComment(book_name='{self.book_name}', author_id='{self.author_id}', star='{self.star}', time='{self.time}', likenum='{self.likenum}', body='{self.body}')"

    def append(self, text):
        self.body += text


trash = []
default_file = "data/doubanbook_top250_comments.txt"

book_comments = {}


# 读取文件
def handle_file(filename=default_file, log=False):
    comment = None
    global trash, book_comments
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过标题行
            # 处理每一行数据
            parts = line.strip().split("\t")
            if len(parts) == 6:
                book_name, author_id, star, time, likenum, body = parts
                if book_name not in book_comments:
                    book_comments[book_name] = []
                comment = BookComment(book_name, author_id, star, time, likenum, body)
                book_comments[book_name].append(comment)
            else:
                if line.strip() and len(parts) < 2 and comment:
                    comment.append(line.strip())
                else:
                    trash.append(line.strip() + "=None")

        # 打印错误行数量
        print(f"Error comments lines: {len(trash)}")  # 480
        all_comments = [
            comment for comments in book_comments.values() for comment in comments
        ]
        # 统计评论数量
        print(f"Total number of comments: {len(all_comments)}")

        # 打印前10条评论
        for comment in all_comments[:10]:
            if log:
                print(comment.__repr__())

        return all_comments


# 写入文件
def write_file(filename="dist/doubanbook_top250_comments.txt", comments=[]):
    with open(filename, "w", encoding="utf-8") as file:
        # 写入标题行
        file.write("book_name\tauthor_id\tstar\ttime\tlikenum\tbody\n")
        for comment in comments:
            line = f"{comment.book_name}\t{comment.author_id}\t{comment.star}\t{comment.time}\t{comment.likenum}\t{comment.body}\n"
            file.write(line)
    print(f"数据已成功写入 {filename}")


if __name__ == "__main__":
    comments = handle_file(log=False)
    for text in trash[:36]:
        if text != "=None":
            print(text)
    # write_file(comments=comments)

    print(comments[0].body)
    print(comments[0].__repr__())
