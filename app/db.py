from typing import final

from peewee import DateTimeField, IntegerField, Model, TextField
from playhouse.sqlite_ext import AutoIncrementField, JSONField, SqliteExtDatabase

from app.settings import settings

db = SqliteExtDatabase(settings.DB_FILE)


@final
class EntityTable(Model):
    id = AutoIncrementField(primary_key=True)
    data = JSONField()

    @final
    class Meta:
        database = db
        table_name = "entities"


@final
class AssignmentTable(Model):
    judge_id = TextField(primary_key=True)
    entity_id_1 = IntegerField()
    entity_id_2 = IntegerField()
    timestamp = DateTimeField()

    @final
    class Meta:
        database = db
        table_name = "assignments"


@final
class SnapshotTable(Model):
    id = AutoIncrementField(primary_key=True)
    timestamp = DateTimeField()
    bdp = JSONField()

    @final
    class Meta:
        database = db
        table_name = "snapshots"


@final
class WriteAheadTable(Model):
    id = AutoIncrementField(primary_key=True)
    timestamp = DateTimeField()
    event = TextField()
    params = JSONField()

    @final
    class Meta:
        database = db
        table_name = "writeahead"
