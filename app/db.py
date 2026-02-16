from peewee import Model, TextField, DateTimeField
from playhouse.sqlite_ext import SqliteExtDatabase, JSONField, AutoIncrementField
from app.constants import DB_FILE

db = SqliteExtDatabase(DB_FILE)


class EntityTable(Model):
    id = AutoIncrementField(primary_key=True)
    data = JSONField()

    class Meta:
        database = db
        table_name = "entities"


class AssignmentTable(Model):
    judge_id = TextField(primary_key=True)
    entity_id_1 = TextField()
    entity_id_2 = TextField()
    timestamp = DateTimeField()

    class Meta:
        database = db
        table_name = "assignments"


class SnapshotTable(Model):
    id = AutoIncrementField(primary_key=True)
    timestamp = DateTimeField()
    state = JSONField()

    class Meta:
        database = db
        table_name = "snapshots"


class WriteAheadTable(Model):
    id = AutoIncrementField(primary_key=True)
    timestamp = DateTimeField()
    event = TextField()
    params = JSONField()

    class Meta:
        database = db
        table_name = "writeahead"
