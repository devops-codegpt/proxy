#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/5 10:46
# @Author  : Jack
# @File    : main.py
# @Software: PyCharm
from typing import Dict

from gerrit import Gerrit
import reviewer


def parse_message(review_msg: str) -> Dict:
    """
    Parse review message and construct structures that can be used by the Gerrit api.
    :param review_msg: review message
    """
    msg_list = review_msg.split(":")
    return {
        "line": msg_list[1].strip(),
        "message": msg_list[3].strip()
    }


def main():
    gerrit = Gerrit(host="http://gerrit-example.com", username="admin", password="password")
    # Get change files from change-number. The following is an example change number
    change_number = 1080417
    files = gerrit.get_revision_files(change_number)

    comments = {}
    for file in files:
        # Get the content of the change file
        change_type, content = gerrit.get_file_content(change_number, file)
        # Code review only for files that have been modified or added
        if change_type not in ["MODIFIED", "ADDED"]:
            continue
        # Start codereview
        review_messages = reviewer.chatgpt_review(filename=file, content=content)
        if not review_messages:
            continue
        file_messages = map(parse_message, review_messages)
        comments[file] = list(file_messages)

    # The composition payload is used to call the gerrit api for codereview
    if not comments:
        return
    payload = {
        "labels": {
            "Lint-Verified": -1,
        },
        "comments": comments
    }
    # print(payload)
    # Set code review for this change number
    gerrit.set_review(change_number=change_number, payload=payload)


if __name__ == '__main__':
    main()
