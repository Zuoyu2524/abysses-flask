CREATE TABLE Results(
  result_id INTEGER PRIMARY KEY,
  image TEXT NOT NULL,
  label TEXT NOT NULL,
  MajorityVoting TEXT
);

CREATE TABLE Feedback(
 feedback_id INTEGER PRIMARY KEY,
 label NOT NULL,
 feedback TEXT,
 result_id INTEGER NOT NULL,
 FOREIGN KEY(result_id) REFERENCES Results(result_id)
);

