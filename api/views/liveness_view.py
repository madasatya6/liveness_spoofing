from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from api.usecases.liveness_usecase import LivenessUsecase


@api_view(['POST'])
def liveness_spoofing_score(request):

    b64 = request.data.get("video_base64")

    if not b64:
        return Response({
            "status": "error",
            "message": "Missing video_base64"
        }, status=status.HTTP_400_BAD_REQUEST)

    usecase = LivenessUsecase()
    result = usecase.process(b64)

    return Response({
        "status": "ok",
        **result
    })
