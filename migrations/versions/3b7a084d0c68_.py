"""empty message

Revision ID: 3b7a084d0c68
Revises: a366358cae68
Create Date: 2024-03-06 15:59:52.435519

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '3b7a084d0c68'
down_revision = 'a366358cae68'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('image_data', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_id', sa.Integer(), nullable=False))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('image_data', schema=None) as batch_op:
        batch_op.drop_column('user_id')

    # ### end Alembic commands ###
