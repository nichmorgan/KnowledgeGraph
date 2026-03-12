from polyfactory.factories.pydantic_factory import ModelFactory
from app import models


class ConceptualKnowledgeFactory(ModelFactory[models.ConceptualKnowledge]):
    @classmethod
    def children(cls) -> list[models.ProceduralKnowledge]:
        return [
            *ProceduralKnowledgeFactory.batch(cls.__random__.randint(1, 3)),
            *AssessmentKnowledgeFactory.batch(cls.__random__.randint(1, 2)),
        ]


class ProceduralKnowledgeFactory(ModelFactory[models.ProceduralKnowledge]):
    @classmethod
    def child(cls) -> models.ProceduralKnowledge | None:
        if cls.__random__.choices([False, True], weights=[0.7, 0.3])[0]:
            return cls.build()
        return None


class AssessmentKnowledgeFactory(ModelFactory[models.AssessmentKnowledge]): ...


class RootKnowledgeFactory(ModelFactory[models.RootKnowledge]):
    @classmethod
    def children(cls) -> list[models.ConceptualKnowledge]:
        concepts = ConceptualKnowledgeFactory.batch(
            cls.__random__.randint(2, 3), connections=[]
        )

        for i, concept in enumerate(concepts):
            others = concepts[:i] + concepts[i + 1 :]
            if to := cls.__random__.choice(others) if others else None:
                to_relations_with_this = [
                    conn.relation for conn in to.connections if conn.to == concept.id
                ]
                this_relations_with_to = [
                    v.value
                    for v in models.KnowledgeConceptualLinkType
                    if v.value not in to_relations_with_this
                ]

                if this_relations_with_to:
                    concept.connections.append(
                        models.ConceptualKnowledgeConnection(
                            relation=cls.__random__.choice(this_relations_with_to),
                            to=to.id,
                        )
                    )
        return concepts
