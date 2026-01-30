from peewee import Model, TextField, DateTimeField
from playhouse.sqlite_ext import SqliteExtDatabase, JSONField, AutoIncrementField
from app.constants import DB_FILE

db = SqliteExtDatabase(DB_FILE)


class EntityTable(Model):
    id = AutoIncrementField()
    data = JSONField()

    class Meta:
        database = db


class AssignmentTable(Model):
    judge_id = TextField()
    project_id_1 = TextField()
    project_id_2 = TextField()
    time = DateTimeField()

    class Meta:
        database = db


class SnapshotTable(Model):
    id = AutoIncrementField()
    time = DateTimeField()
    state = JSONField()

    class Meta:
        database = db


class LogTable(Model):
    id = AutoIncrementField()
    time = DateTimeField()
    event = TextField()
    params = JSONField()

    class Meta:
        database = db
