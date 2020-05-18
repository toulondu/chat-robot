# chat-robot
从零开始搞一个聊天机器人

## 数据
数据1：[Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)：电影剧本数据，包含来自617部电影中9035个角色间的30万次发言
数据集中含有readme文件，用以对数据集中各文件的结构予以说明。
其中主要的2个文件的结构如下：
- movie_lines.txt
	- contains the actual text of each utterance
	- fields:
		- lineID
		- characterID (who uttered this phrase)
		- movieID
		- character name
		- text of the utterance

- movie_conversations.txt
	- the structure of the conversations
	- fields
		- characterID of the first character involved in the conversation
		- characterID of the second character involved in the conversation
		- movieID of the movie in which the conversation occurred
		- list of the utterances that make the conversation, in chronological 
			order: ['lineID1','lineID2',�,'lineIDN']
			has to be matched with movie_lines.txt to reconstruct the actual content
