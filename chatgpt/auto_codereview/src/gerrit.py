#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/6 11:03
# @Author  : Jack
# @File    : gerrit.py
# @Software: PyCharm
from typing import Iterable, Any, Dict, Tuple

from pGerrit.client import GerritClient
from requests.auth import HTTPBasicAuth


class Gerrit(object):
    """Class connects to Gerrit and perform related operations on Gerrit"""

    def __init__(self, host: str, username: str, password: str):
        auth = HTTPBasicAuth(username, password)
        self.client = GerritClient(host=host, auth=auth)

    def get_revision_files(self, change_number: int) -> Iterable:
        """
        Retrieve the files of the change revision.
        :param change_number: Gerrit change number
        :return: change files id
        """
        change = self.client.change(f"{change_number}")
        current_revision = change.current_revision()
        for f in vars(current_revision.files()):
            file_id = str(f).strip()
            if file_id == "/COMMIT_MSG":
                continue
            yield file_id

    def get_file_content(self, change_number: int, file_id: str) -> Tuple[str, str]:
        """
        Get the content of a file under a change.
        :param change_number: change number
        :param file_id: file id
        :return: file content
        """
        change = self.client.change(f"{change_number}")
        current_revision = change.current_revision()
        file_diff = current_revision.file(file_id).diff()
        file_diff = vars(file_diff)
        change_type = file_diff.get("change_type", "")
        change_contents = file_diff.get("content", [])
        contents = []
        for c in change_contents:
            c = vars(c)
            if c.get("ab", []):
                contents.extend(c["ab"])
            if c.get("b", []):
                contents.extend(c["b"])
        # Add line count markers to the code
        line_contents = []
        for i, v in enumerate(contents):
            line_contents.append("{:>4d} {}".format(i + 1, v))
        return change_type, "\n".join(line_contents)

    def set_review(self, change_number: int, payload: Dict[str, Any]) -> None:
        """
         Set a review for the change revision.
        :param change_number: change number
        :param payload: review body `ReviewInput <https://gerrit-review.googlesource.com/Documentation/rest-api-changes.html#review-input>`

        Usage:
            payload = {
                "labels": {
                    "Lint-Verified": +1,
                },
                "comments": {
                    "proxy/chatgpt/src/qabot/main.py": [
                        {
                            "message": "a good path method use",
                            "line": 30,
                        }
                    ]
                },
            }

            result = set_review(1080417, payload)

        """
        change = self.client.change(f"{change_number}")
        current_revision = change.current_revision()
        result = current_revision.set_review(payload)
        return result
