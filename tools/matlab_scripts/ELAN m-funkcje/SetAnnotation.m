function oElan = SetAnnotation(p_oElan, p_sTierName, p_iAnnotationIndex, p_sFields)

%Ustawia pola adnotacji na pozycji p_iAnnotationIndex z warstwy o 
%nazwie p_sTierName obiektu p_oElan. Warto�ci p�l okre�la parametr 
%p_sFields.
%
%parametry:
%
%p_oElan - obiekt, na kt�rego danych ma operowa� funkcja. Parametr 
%ten musi by� obiektem struktury oElan. W przeciwnym wypadku funkcja
%zostaje przerwana z b��dem.
%
%p_sTierName - nazwa warstwy, z kt�rej adnotacja ma zosta� 
%zmodyfikowana. Warstwa o danej nazwie musi istnie�. Parametr ten musi
%by� typu string. W przeciwnym wypadku funkcja zostaje przerwana 
%z b��dem.
%
%p_iAnnotationIndex - indeks okre�laj�cy pozycj� adnotacji, kt�ra ma 
%zosta� zmodyfikowana. Parametr ten musi przyjmowa� warto�� niewi�ksz�
%ni� liczba wszystkich adnotacji w danej warstwie. Musi by� r�wnie� 
%typu numerycznego. W przeciwnym wypadku funkcja zostaje przerwana 
%z b��dem.
%
%p_sFields - tablica kom�rek typu string przechowuj�ca nowe warto�ci 
%p�l, kt�re maj� zosta� ustawione w adnotacji. Parametr ten musi by� 
%typu cell array of strings. W przeciwnym wypadku funkcja zostaje 
%przerwana z b��dem.
%
%warto�� zwracana:
%
%Zaktualizowany obiekt struktury oElan.

	if isstruct(p_oElan) == 0
		oElan = [];	
		disp('Error: p_oElan is not a structure');
		return;
	end
	
	if ~isfield(p_oElan, 'oTiers') || ~isfield(p_oElan.oTiers, 'oAnnotations') || ~isfield(p_oElan.oTiers, 'sName')
		oElan = [];
		disp('Error: p_oElan is a wrong structure');
		return;
	end
	
	if ischar(p_sTierName) == 0 
		oElan = p_oElan;
		disp('Error: p_sTierName is not a string.');
		return;
	end
	
	if isnumeric(p_iAnnotationIndex) == 0 
		oElan = p_oElan;
		disp('Error: p_iAnnotationIndex is not a number.');
		return;
	end
	
	if iscellstr(p_sFields) == 0 
		oElan = p_oElan;
		disp('Error: p_sFields is not a cell array of strings.');
		return;
	end
	
	bFound = false;
	for i = 1 : numel(p_oElan.oTiers)
		sTemp = p_oElan.oTiers(i).sName;
		if strcmp(p_sTierName, sTemp)
			
			if p_iAnnotationIndex > numel(p_oElan.oTiers(i).oAnnotations) || p_iAnnotationIndex < 1
				oElan = p_oElan;
				disp('Error: Annotation of such index does not exist.');
				return;
			end
			
			p_oElan.oTiers(i).oAnnotations(p_iAnnotationIndex).sFields = p_sFields;
			
			bFound = true;
		end
	end
	
	if ~bFound
		oElan = p_oElan;
		disp('Error: Tier of such name does not exist.');
		return;
	end
	
	oElan = p_oElan;
	
end